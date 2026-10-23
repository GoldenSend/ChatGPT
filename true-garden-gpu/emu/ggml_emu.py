"""Bit-exact numpy emulation of the llama.cpp CPU oracle for one Qwen3 layer.

Target: llama.cpp commit 23b0202a189c44a54625aadcb37a946dd1d6278d (the PS4
CPU baseline), ggml-cpu built for AMD Jaguar (-march=btver2: SSE4.2, AVX,
F16C; no AVX2, no FMA). Every function below mirrors the exact sequence of
float32 roundings of the kernel that the Jaguar build selects, including
SIMD lane assignment and horizontal-reduction order, because the CPU oracle
contains discontinuous steps (Q8_K activation quantization, FP16 rounding of
Q / K / V and of the flash-attention accumulator) that turn last-bit
differences into whole quantization steps.

Kernel sources mirrored (paths relative to llama.cpp/ggml/src):
  quantize_row_q8_K_ref ............ ggml-quants.c
  ggml_vec_dot_q4_K_q8_K (__AVX__) . ggml-cpu/arch/x86/quants.c
  ggml_vec_dot_q6_K_q8_K (__AVX__) . ggml-cpu/arch/x86/quants.c
  ggml_vec_dot_f16 (AVX+F16C) ...... ggml-cpu/vec.cpp + simd-mappings.h
  rms_norm_f32 ..................... ggml-cpu/ops.cpp (double accumulation)
  rope (NEOX) ...................... ggml-cpu/ops.cpp (theta chain, cosf/sinf)
  flash_attn_ext_f16 (F16 V) ....... ggml-cpu/ops.cpp (FP16 VKQ accumulator)
  soft_max_f32 ..................... ggml-cpu/ops.cpp + vec.cpp (SSE2 ggml_v_expf)
  swiglu ........................... ggml-cpu/vec.cpp (SSE2 ggml_v_silu)

libm (expf, cosf, sinf, powf) is called through ctypes so the emulator uses
the same glibc the oracle used. The PS4 runs Debian 13 glibc; this container
runs Ubuntu 24.04 glibc 2.39. Both use the post-2.28 single-precision
implementations, but that equivalence is an assumption until checked on the
PS4 (see TESTS/README.md).
"""
import ctypes
import ctypes.util

import numpy as np

F32 = np.float32
QK_K = 256

_libm = ctypes.CDLL(ctypes.util.find_library("m"))
for _fn in ("expf", "cosf", "sinf", "powf", "sqrtf"):
    getattr(_libm, _fn).restype = ctypes.c_float
_libm.expf.argtypes = [ctypes.c_float]
_libm.cosf.argtypes = [ctypes.c_float]
_libm.sinf.argtypes = [ctypes.c_float]
_libm.sqrtf.argtypes = [ctypes.c_float]
_libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]


def expf(x):
    return F32(_libm.expf(float(x)))


def f16r(x):
    """FP32 -> FP16 -> FP32 with round-to-nearest-even (F16C _cvtps_ph mode 0)."""
    return np.asarray(x, dtype=F32).astype(np.float16).astype(F32)


# --------------------------------------------------------------------------
# block decoding

def _f16_at(raw, off):
    return raw[..., off:off + 2].copy().view(np.float16)[..., 0].astype(F32)


def q4k_unpack(raw):
    """raw: uint8 [rows, nb, 144] -> d, dmin [rows,nb] f32; sc, mn [rows,nb,8] int; q [rows,nb,256] int (0..15)."""
    d = _f16_at(raw, 0)
    dmin = _f16_at(raw, 2)
    s = raw[..., 4:16].astype(np.int64)
    sc = np.empty(raw.shape[:-1] + (8,), np.int64)
    mn = np.empty_like(sc)
    for j in range(8):  # get_scale_min_k4
        if j < 4:
            sc[..., j] = s[..., j] & 63
            mn[..., j] = s[..., j + 4] & 63
        else:
            sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
            mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    qs = raw[..., 16:144].astype(np.int64).reshape(raw.shape[:-1] + (4, 32))
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(raw.shape[:-1] + (256,))
    return d, dmin, sc, mn, q


def q6k_unpack(raw):
    """raw: uint8 [rows, nb, 210] -> d [rows,nb]; sc [rows,nb,16] int; q [rows,nb,256] int (0..63, no -32)."""
    ql = raw[..., 0:128].astype(np.int64)
    qh = raw[..., 128:192].astype(np.int64)
    sc = raw[..., 192:208].view(np.int8).astype(np.int64)
    d = _f16_at(raw, 208)
    q = np.empty(raw.shape[:-1] + (256,), np.int64)
    for n in range(2):
        L = ql[..., 64 * n:64 * n + 64]
        H = qh[..., 32 * n:32 * n + 32]
        b = 128 * n
        q[..., b + 0:b + 32] = (L[..., 0:32] & 0xF) | (((H >> 0) & 3) << 4)
        q[..., b + 32:b + 64] = (L[..., 32:64] & 0xF) | (((H >> 2) & 3) << 4)
        q[..., b + 64:b + 96] = (L[..., 0:32] >> 4) | (((H >> 4) & 3) << 4)
        q[..., b + 96:b + 128] = (L[..., 32:64] >> 4) | (((H >> 6) & 3) << 4)
    return d, sc, q


def dequant_q6k_row(raw_row):
    """dequantize_row_q6_K for one row: raw [nb, 210] -> f32 [nb*256]."""
    d, sc, q = q6k_unpack(raw_row)
    q = q - 32
    scale = np.repeat(sc, 16, axis=-1).astype(F32)  # sub-block of 16
    return ((d[..., None] * scale) * q.astype(F32)).astype(F32).reshape(-1)


# --------------------------------------------------------------------------
# Q8_K activation quantization (the main discontinuity)

def quantize_q8k(x):
    """quantize_row_q8_K_ref. x: f32 [n] -> (d [nb] f32, qs [nb,256] int, bsums [nb,16] int)."""
    x = np.asarray(x, F32).reshape(-1, QK_K)
    ax = np.abs(x)
    imax = np.argmax(ax, axis=1)  # first index of the largest |x| (strict '>' scan)
    mx = x[np.arange(x.shape[0]), imax]
    amax = ax[np.arange(x.shape[0]), imax]
    zero = amax == 0
    mx_safe = np.where(zero, F32(1), mx).astype(F32)
    iscale = (F32(-127.0) / mx_safe).astype(F32)
    v = (iscale[:, None] * x).astype(F32)
    qs = np.rint(v).astype(np.int64)  # nearest_int: +1.5*2^23 trick == RNE
    qs = np.minimum(127, qs)
    qs[zero] = 0
    d = (F32(1.0) / iscale).astype(F32)
    d[zero] = 0
    bsums = qs.reshape(-1, 16, 16).sum(axis=2)
    return d, qs, bsums


# --------------------------------------------------------------------------
# horizontal reductions

def _hsum_float_8(acc):
    """hsum_float_8: acc [..., 8] f32."""
    r = (acc[..., 4:8] + acc[..., 0:4]).astype(F32)
    r2 = (r[..., 0:2] + r[..., 2:4]).astype(F32)  # _mm_movehl_ps
    return (r2[..., 0] + r2[..., 1]).astype(F32)  # _mm_movehdup_ps + add_ss


def _hsum_4_movehl(a):
    """acc_m reduction: [a0+a2, a1+a3] then (a0+a2)+(a1+a3)."""
    r = (a[..., 0:2] + a[..., 2:4]).astype(F32)
    return (r[..., 0] + r[..., 1]).astype(F32)


# --------------------------------------------------------------------------
# quantized mat-vec (one activation column)

def matvec_q4k(raw, x):
    """ggml_vec_dot_q4_K_q8_K (__AVX__ path) for every row. raw: uint8 [rows, nb, 144]."""
    yd, yq, yb = quantize_q8k(x)
    d16, dmin16, sc, mn, q = q4k_unpack(raw)
    rows, nb = raw.shape[0], raw.shape[1]
    # products per element, laid out [rows, nb, j(4), lh(2), h(2), k(4), e(4)]
    prod = (q * yq[None]).reshape(rows, nb, 4, 2, 2, 4, 4).sum(axis=-1)
    scl = sc.reshape(rows, nb, 4, 2)[..., None, None]
    sumi = (prod * scl).sum(axis=(2, 3)).reshape(rows, nb, 8)  # lane = h*4 + k
    # mins: madd(mins[8], hadd(bsums)) -> 4 lanes
    q8s = yb.reshape(nb, 8, 2).sum(axis=-1)  # [nb, 8]
    pm = (mn * q8s[None]).reshape(rows, nb, 4, 2).sum(axis=-1)  # [rows, nb, 4]
    acc = np.zeros((rows, 8), F32)
    acc_m = np.zeros((rows, 4), F32)
    for i in range(nb):
        d = (yd[i] * d16[:, i]).astype(F32)
        dmin = (-yd[i] * dmin16[:, i]).astype(F32)
        acc_m = ((dmin[:, None] * pm[:, i].astype(F32)).astype(F32) + acc_m).astype(F32)
        acc = ((d[:, None] * sumi[:, i].astype(F32)).astype(F32) + acc).astype(F32)
    return (_hsum_float_8(acc) + _hsum_4_movehl(acc_m)).astype(F32)


def matvec_q6k(raw, x):
    """ggml_vec_dot_q6_K_q8_K (__AVX__ path) for every row. raw: uint8 [rows, nb, 210]."""
    yd, yq, yb = quantize_q8k(x)
    d16, sc, q = q6k_unpack(raw)
    rows, nb = raw.shape[0], raw.shape[1]
    # chunk c (16 elements, c = 0..15) = j*8 + i ; lane half = i % 2 ; lane k = (e//4)
    prod = (q * yq[None]).reshape(rows, nb, 2, 8, 4, 4).sum(axis=-1)  # [rows,nb,j,i,k]
    scl = sc.reshape(rows, nb, 2, 8)[..., None]
    p = prod * scl  # [rows, nb, j, i, k]
    sumi0 = p[:, :, :, 0::2, :].sum(axis=(2, 3))  # even chunks -> sumi_0
    sumi1 = p[:, :, :, 1::2, :].sum(axis=(2, 3))
    sub = (yb[None] * sc).reshape(rows, nb, 2, 4, 2).sum(axis=-1) * 32  # [rows, nb, half, lane]
    sumi0 = sumi0 - sub[:, :, 0]
    sumi1 = sumi1 - sub[:, :, 1]
    sumi = np.concatenate([sumi0, sumi1], axis=-1)
    acc = np.zeros((rows, 8), F32)
    for i in range(nb):
        d = (yd[i] * d16[:, i]).astype(F32)
        acc = ((d[:, None] * sumi[:, i].astype(F32)).astype(F32) + acc).astype(F32)
    return _hsum_float_8(acc)


# --------------------------------------------------------------------------
# elementwise / normalization

def rms_norm(x, eps):
    """ggml_compute_forward_rms_norm_f32 for the last axis: double accumulation, sequential."""
    x = np.asarray(x, F32)
    sq = (x * x).astype(F32).astype(np.float64)
    s = np.cumsum(sq, axis=-1)[..., -1]  # sequential double sum
    n = x.shape[-1]
    mean = (s / n).astype(F32)
    scale = (F32(1.0) / np.sqrt((mean + F32(eps)).astype(F32))).astype(F32)
    return (x * scale[..., None]).astype(F32)


def rope_neox_cache(pos, n_dims=128, freq_base=1e6):
    theta_scale = F32(_libm.powf(F32(freq_base), F32(-2.0) / F32(n_dims)))
    theta = F32(pos)
    cos = np.empty(n_dims // 2, F32)
    sin = np.empty(n_dims // 2, F32)
    for i in range(n_dims // 2):
        cos[i] = F32(_libm.cosf(theta)) * F32(1.0)
        sin[i] = F32(_libm.sinf(theta)) * F32(1.0)
        theta = F32(theta * theta_scale)
    return cos, sin


def rope_neox(x, pos, n_dims=128, freq_base=1e6, cache=None):
    """x: [heads, 128] f32."""
    cos, sin = cache if cache is not None else rope_neox_cache(pos, n_dims, freq_base)
    h = n_dims // 2
    x0, x1 = x[:, :h], x[:, h:]
    y = np.empty_like(x)
    y[:, :h] = ((x0 * cos).astype(F32) - (x1 * sin).astype(F32)).astype(F32)
    y[:, h:] = ((x0 * sin).astype(F32) + (x1 * cos).astype(F32)).astype(F32)
    return y


def v_expf(x):
    """ggml_v_expf, SSE2 variant without FMA (MADD128 = mul then add)."""
    x = np.asarray(x, F32)
    r = F32(12582912.0)
    z = ((x * F32(float.fromhex('0x1.715476p+0'))).astype(F32) + r).astype(F32)
    n = (z - r).astype(F32)
    b = ((x - (n * F32(float.fromhex('0x1.62e4p-1'))).astype(F32)).astype(F32)
         - (n * F32(float.fromhex('0x1.7f7d1cp-20'))).astype(F32)).astype(F32)
    e = (z.view(np.uint32) << np.uint32(23)).astype(np.uint32)
    k = (e + np.array(1.0, F32).view(np.uint32)).astype(np.uint32).view(F32)
    c = np.abs(n) > F32(126)
    u = (b * b).astype(F32)
    t1 = ((F32(float.fromhex('0x1.0e4020p-7')) * b).astype(F32) + F32(float.fromhex('0x1.573e2ep-5'))).astype(F32)
    t2 = ((F32(float.fromhex('0x1.555e66p-3')) * b).astype(F32) + F32(float.fromhex('0x1.fffdb6p-2'))).astype(F32)
    t3 = ((t1 * u).astype(F32) + t2).astype(F32)
    j = ((t3 * u).astype(F32) + (F32(float.fromhex('0x1.ffffecp-1')) * b).astype(F32)).astype(F32)
    res = ((j * k).astype(F32) + k).astype(F32)
    if np.any(c):
        g = np.where(n <= 0, np.uint32(0x82000000), np.uint32(0)).astype(np.uint32)
        s1 = (g + np.uint32(0x7f000000)).astype(np.uint32).view(F32)
        s2 = (e - g).astype(np.uint32).view(F32)
        dd = np.abs(n) > F32(192)
        alt = np.where(dd, (s1 * s1).astype(F32),
                       (((s2 * j).astype(F32) + s2).astype(F32) * s1).astype(F32))
        res = np.where(c, alt, res).astype(F32)
    return res


def swiglu(g, u):
    """ggml_vec_swiglu_f32, SSE2 path: silu(g) * u with ggml_v_silu (n % 4 == 0 assumed)."""
    g = np.asarray(g, F32)
    assert g.size % 4 == 0
    ex = v_expf((F32(0) - g).astype(F32))
    silu = (g / (F32(1) + ex).astype(F32)).astype(F32)
    return (silu * np.asarray(u, F32)).astype(F32)


# --------------------------------------------------------------------------
# attention

def vec_dot_f16(x16, y16):
    """ggml_vec_dot_f16, AVX+F16C: 4 accumulators x 8 lanes, mul+add, F32x8_REDUCE. n % 32 == 0."""
    x = np.asarray(x16, F32).reshape(-1, 4, 8)
    y = np.asarray(y16, F32).reshape(-1, 4, 8)
    acc = np.zeros((4, 8), F32)
    for i in range(x.shape[0]):
        acc = ((x[i] * y[i]).astype(F32) + acc).astype(F32)
    a0 = (acc[0] + acc[2]).astype(F32)
    a1 = (acc[1] + acc[3]).astype(F32)
    a0 = (a0 + a1).astype(F32)
    t0 = (a0[0:4] + a0[4:8]).astype(F32)
    t1 = np.array([t0[0] + t0[1], t0[2] + t0[3]], F32)
    return F32(t1[0] + t1[1])  # (double) then stored to float *s


def flash_attn_f16(q, k16, v16, scale):
    """ggml_compute_forward_flash_attn_ext_f16_one_chunk with F16 K/V, no mask holes.

    q: [n_head, D] f32 (one query token); k16, v16: [n_kv, n_head_kv, D] (values already FP16-exact f32).
    Returns [n_head, D] f32.
    """
    n_head, D = q.shape
    n_kv, n_head_kv, _ = k16.shape
    rk = n_head // n_head_kv
    out = np.empty((n_head, D), F32)
    for h in range(n_head):
        kvh = h // rk
        qq = f16r(q[h])
        S = F32(0.0)
        M = F32(-np.inf)
        vkq = np.zeros(D, F32)  # holds FP16 values
        for ic in range(n_kv):
            s = vec_dot_f16(k16[ic, kvh], qq)
            s = F32(s * scale)
            s = F32(s + F32(0.0))  # mask value 0 * slope 1
            Mold = M
            ms = F32(1.0)
            vs = F32(1.0)
            if s > M:
                M = s
                ms = expf(Mold - M)
                vkq = f16r(vkq * ms)
            else:
                vs = expf(s - M)
            vkq = f16r((vkq + (v16[ic, kvh] * vs).astype(F32)).astype(F32))
            S = F32(F32(S * ms) + vs)
        sinv = F32(0.0) if S == 0 else F32(F32(1.0) / S)
        out[h] = (vkq * sinv).astype(F32)
    return out


def attn_nofa(q, k16, v16, scale):
    """Non-flash path: kq = mul_mat(K f16, Q->f16); soft_max_ext(scale); kqv = mul_mat(V^T f16, softmax->f16).

    Assumes the mask leaves exactly n_kv valid slots and that the remaining
    (masked) slots of the n_ctx-wide row contribute exact zeros; n_ctx must be
    passed through q-agnostic padding below.
    """
    raise NotImplementedError("non-FA path: see NUMERICAL-ANALYSIS.md, not yet emulated")


# --------------------------------------------------------------------------
# one Qwen3 layer, one token, KV cache carried across calls

class Qwen3Layer0:
    OPS = ("rms_norm", "matvec_q4k", "matvec_q6k", "rope_neox_cache", "rope_neox", "f16r",
           "flash_attn_f16", "swiglu")

    def __init__(self, w, n_head=16, n_head_kv=8, head_dim=128, eps=1e-6, freq_base=1e6, ops=None):
        """ops: optional {name: replacement} for any of OPS (used to model non-ggml GPU arithmetic)."""
        self.w = w
        g = globals()
        self.op = {k: (ops or {}).get(k, g[k]) for k in self.OPS}
        self.n_head, self.n_head_kv, self.hd = n_head, n_head_kv, head_dim
        self.eps, self.freq_base = eps, freq_base
        self.kq_scale = F32(F32(1.0) / F32(_libm.sqrtf(F32(head_dim))))
        self.k16 = []
        self.v16 = []

    def step(self, x, pos, hooks=None):
        """x: f32 [1024] residual input (embedding row). Returns dict of named intermediates.

        hooks: optional {name: fn(array) -> array} applied to an intermediate
        right after it is computed (used by the sensitivity experiments).
        """
        w, t = self.w, {}

        def put(name, val):
            if hooks and name in hooks:
                val = hooks[name](val)
            t[name] = val
            return val

        op = self.op
        x = put("inp", np.asarray(x, F32))
        a = put("attn_norm", (op["rms_norm"](x, self.eps) * w["attn_norm"]).astype(F32))
        q = put("Qcur", op["matvec_q4k"](w["attn_q"], a))
        k = put("Kcur", op["matvec_q4k"](w["attn_k"], a))
        v = put("Vcur", op["matvec_q6k"](w["attn_v"], a))
        qn = put("Qcur_normed", (op["rms_norm"](q.reshape(self.n_head, self.hd), self.eps) * w["attn_q_norm"]).astype(F32))
        kn = put("Kcur_normed", (op["rms_norm"](k.reshape(self.n_head_kv, self.hd), self.eps) * w["attn_k_norm"]).astype(F32))
        cache = op["rope_neox_cache"](pos, self.hd, self.freq_base)
        qr = put("Qrope", op["rope_neox"](qn, pos, cache=cache))
        kr = put("Krope", op["rope_neox"](kn, pos, cache=cache))
        self.k16.append(op["f16r"](kr))
        self.v16.append(op["f16r"](v.reshape(self.n_head_kv, self.hd)))
        o = put("kqv_out", op["flash_attn_f16"](qr, np.stack(self.k16), np.stack(self.v16), self.kq_scale).reshape(-1))
        ao = put("attn_out", op["matvec_q4k"](w["attn_output"], o))
        fi = put("ffn_inp", (ao + x).astype(F32))
        fn = put("ffn_norm", (op["rms_norm"](fi, self.eps) * w["ffn_norm"]).astype(F32))
        g = put("ffn_gate", op["matvec_q4k"](w["ffn_gate"], fn))
        u = put("ffn_up", op["matvec_q4k"](w["ffn_up"], fn))
        s = put("ffn_swiglu", op["swiglu"](g, u))
        fo = put("ffn_out", op["matvec_q6k"](w["ffn_down"], s))
        put("l_out", (fo + fi).astype(F32))
        return t


def load_weights(gguf_path, llama_cpp=None):
    """Read layer-0 tensors (raw quant blocks) from a GGUF via gguf-py."""
    import os
    import sys
    if llama_cpp:
        sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    import gguf

    r = gguf.GGUFReader(gguf_path)
    out = {}
    for t in r.tensors:
        name = t.name.replace("blk.0.", "").replace(".weight", "")
        typ = gguf.GGMLQuantizationType(t.tensor_type).name
        data = np.asarray(t.data)
        ne = [int(v) for v in t.shape]  # ggml ne order
        if typ == "F32":
            out[name] = data.astype(F32).reshape(-1) if len(ne) == 1 or ne[1] == 1 else data.astype(F32)
        elif typ in ("Q4_K", "Q6_K"):
            bsz = 144 if typ == "Q4_K" else 210
            rows = ne[1]
            out[name] = data.view(np.uint8).reshape(rows, ne[0] // QK_K, bsz)
        else:
            raise ValueError(f"{t.name}: unsupported type {typ}")
        out[name + ".type"] = typ
    return out
