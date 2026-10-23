#!/usr/bin/env python3
"""Which GPU-plausible arithmetic choices make token 1 fail while token 0 passes?

Runs the bit-exact oracle emulator (emu/ggml_emu.py) next to variants in
which one op is computed the way a careful GPU shader typically would:
numerically sound, often *more* accurate, but not the same rounding
sequence as ggml on Jaguar. For each variant it records, per decode step,
the first stage that differs, Q8_K code flips at every activation
quantization site, FP16 flips in the K/V cache, and how many final layer
outputs (l_out) exceed several tolerances.

usage: sensitivity.py OUT.json MODEL.gguf [MODEL.gguf ...]
Weights are synthetic (see tools/make_synth_qwen3.py); magnitudes are not
the real model's, so compare patterns and ratios, not absolute counts.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "emu"))
import ggml_emu as E  # noqa: E402

F32 = np.float32
STAGES = ["attn_norm", "Qcur", "Kcur", "Vcur", "Qcur_normed", "Kcur_normed", "Qrope", "Krope",
          "kqv_out", "attn_out", "ffn_inp", "ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_out", "l_out"]
Q8_SITES = {"attn_norm": "QKV in", "kqv_out": "O-proj in", "ffn_norm": "gate/up in", "ffn_swiglu": "down in"}
TOLS = [(1e-6, 0.0), (1e-5, 0.0), (1e-4, 0.0), (0.0, 1e-3)]  # (atol, rtol)


# ---------------------------------------------------------------- variants

def rms_norm_f32_tree(x, eps):
    """GPU-style: squares summed in float32 by a 256-wide workgroup tree reduction."""
    x = np.asarray(x, F32)
    n = x.shape[-1]
    sq = (x * x).astype(F32)
    width = min(256, n)
    part = sq.reshape(sq.shape[:-1] + (n // width, width)).sum(axis=-2, dtype=F32)  # per-thread serial
    while part.shape[-1] > 1:
        h = part.shape[-1] // 2
        part = (part[..., :h] + part[..., h:]).astype(F32)
    mean = (part[..., 0] / F32(n)).astype(F32)
    scale = (F32(1.0) / np.sqrt((mean + F32(eps)).astype(F32))).astype(F32)
    return (x * scale[..., None]).astype(F32)


def matvec_q4k_rowserial(raw, x):
    yd, yq, yb = E.quantize_q8k(x)
    d16, dmin16, sc, mn, q = E.q4k_unpack(raw)
    rows, nb = raw.shape[:2]
    isum = ((q * yq[None]).reshape(rows, nb, 8, 32).sum(-1) * sc).sum(-1)
    msum = (mn * yb.reshape(nb, 8, 2).sum(-1)[None]).sum(-1)
    acc = np.zeros(rows, F32)
    for i in range(nb):
        d = (yd[i] * d16[:, i]).astype(F32)
        dm = (yd[i] * dmin16[:, i]).astype(F32)
        acc = (acc + ((d * isum[:, i].astype(F32)).astype(F32) - (dm * msum[:, i].astype(F32)).astype(F32))).astype(F32)
    return acc


def matvec_q6k_rowserial(raw, x):
    yd, yq, yb = E.quantize_q8k(x)
    d16, sc, q = E.q6k_unpack(raw)
    rows, nb = raw.shape[:2]
    isum = (((q - 32) * yq[None]).reshape(rows, nb, 16, 16).sum(-1) * sc).sum(-1)
    acc = np.zeros(rows, F32)
    for i in range(nb):
        d = (yd[i] * d16[:, i]).astype(F32)
        acc = (acc + (d * isum[:, i].astype(F32)).astype(F32)).astype(F32)
    return acc


def rope_cache_direct(pos, n_dims=128, freq_base=1e6):
    """GPU-style: theta_i = pos * base^(-2i/n) evaluated directly (float64 here, i.e. best case), trig rounded once."""
    i = np.arange(n_dims // 2, dtype=np.float64)
    theta = (pos * freq_base ** (-2.0 * i / n_dims)).astype(F32).astype(np.float64)
    return np.cos(theta).astype(F32), np.sin(theta).astype(F32)


def rope_cache_gpu_trig(pos, n_dims=128, freq_base=1e6, _rng=np.random.default_rng(7)):
    """Model of hardware sin/cos: correctly rounded value perturbed by +-1 ulp at random (assumption, not measured)."""
    c, s = rope_cache_direct(pos, n_dims, freq_base)
    jit = lambda v: np.nextafter(v, np.where(_rng.random(v.shape) < 0.5, -np.inf, np.inf).astype(F32)).astype(F32)
    if pos == 0:
        return c, s  # sin(0)=0, cos(0)=1 are exact on any sane implementation
    return jit(c), jit(s)


def exp_cr(x):
    return F32(np.exp(np.float64(x)))


def flash_attn_f16_crexp(q, k16, v16, scale):
    """ggml FA sequence, but expf is correctly rounded instead of glibc's."""
    saved = E.expf
    E.expf = exp_cr
    try:
        return E.flash_attn_f16(q, k16, v16, scale)
    finally:
        E.expf = saved


def attn_fp32_standard(q, k16, v16, scale):
    """Typical GPU attention: Q kept fp32, two-pass softmax in fp32, fp32 weighted sum, then divide."""
    n_head, D = q.shape
    n_kv, n_head_kv, _ = k16.shape
    rk = n_head // n_head_kv
    out = np.empty((n_head, D), F32)
    for h in range(n_head):
        K = k16[:, h // rk].astype(F32)
        V = v16[:, h // rk].astype(F32)
        s = np.array([np.dot(K[i].astype(np.float64), q[h].astype(np.float64)) for i in range(n_kv)]).astype(F32)
        s = (s * scale).astype(F32)
        m = s.max()
        p = np.array([exp_cr(v - m) for v in s], F32)
        S = F32(p.sum(dtype=F32))
        acc = np.zeros(D, F32)
        for i in range(n_kv):
            acc = (acc + (V[i] * p[i]).astype(F32)).astype(F32)
        out[h] = (acc / S).astype(F32)
    return out


def _fa_variant(q, k16, v16, scale, q_f16=True, dot="ggml", acc_f16=True):
    """ggml FA sequence with exactly one sub-step changed (for isolating what must match)."""
    n_head, D = q.shape
    n_kv, n_head_kv, _ = k16.shape
    rk = n_head // n_head_kv
    out = np.empty((n_head, D), F32)
    rnd = E.f16r if acc_f16 else (lambda a: np.asarray(a, F32))
    for h in range(n_head):
        kvh = h // rk
        qq = E.f16r(q[h]) if q_f16 else np.asarray(q[h], F32)
        S, M = F32(0.0), F32(-np.inf)
        vkq = np.zeros(D, F32)
        for ic in range(n_kv):
            if dot == "ggml":
                s = E.vec_dot_f16(k16[ic, kvh], qq)
            else:  # one fp32 accumulator, sequential
                s = F32(0.0)
                for a, b in zip(k16[ic, kvh].astype(F32), qq):
                    s = F32(s + F32(a * b))
            s = F32(F32(s * scale) + F32(0.0))
            ms, vs = F32(1.0), F32(1.0)
            if s > M:
                Mold, M = M, s
                ms = E.expf(Mold - M)
                vkq = rnd(vkq * ms)
            else:
                vs = E.expf(s - M)
            vkq = rnd((vkq + (v16[ic, kvh] * vs).astype(F32)).astype(F32))
            S = F32(F32(S * ms) + vs)
        out[h] = (vkq * F32(F32(1.0) / S)).astype(F32)
    return out


def fa_q_fp32(q, k16, v16, scale):
    return _fa_variant(q, k16, v16, scale, q_f16=False)


def fa_dot_serial(q, k16, v16, scale):
    return _fa_variant(q, k16, v16, scale, dot="serial")


def fa_acc_fp32(q, k16, v16, scale):
    return _fa_variant(q, k16, v16, scale, acc_f16=False)


def swiglu_precise(g, u):
    g = np.asarray(g, F32)
    ex = np.exp(-g.astype(np.float64)).astype(F32)
    return ((g / (F32(1) + ex).astype(F32)).astype(F32) * np.asarray(u, F32)).astype(F32)


VARIANTS = {
    "exact (control)": {},
    "rms_norm fp32 tree": {"rms_norm": rms_norm_f32_tree},
    "matvec row-serial": {"matvec_q4k": matvec_q4k_rowserial, "matvec_q6k": matvec_q6k_rowserial},
    "rope direct theta": {"rope_neox_cache": rope_cache_direct},
    "rope hw-trig model": {"rope_neox_cache": rope_cache_gpu_trig},
    "FA ggml + CR exp": {"flash_attn_f16": flash_attn_f16_crexp},
    "attn fp32 standard": {"flash_attn_f16": attn_fp32_standard},
    "FA: Q kept fp32": {"flash_attn_f16": fa_q_fp32},
    "FA: KQ dot serial": {"flash_attn_f16": fa_dot_serial},
    "FA: fp32 accumulator": {"flash_attn_f16": fa_acc_fp32},
    "silu precise exp": {"swiglu": swiglu_precise},
    "all precise": {"rms_norm": rms_norm_f32_tree, "matvec_q4k": matvec_q4k_rowserial,
                    "matvec_q6k": matvec_q6k_rowserial, "rope_neox_cache": rope_cache_direct,
                    "flash_attn_f16": attn_fp32_standard, "swiglu": swiglu_precise},
}


# ---------------------------------------------------------------- metrics

def bits(a):
    return np.asarray(a, F32).reshape(-1).view(np.uint32)


def compare(ref, got):
    r = {}
    first = None
    for st in STAGES:
        nd = int(np.sum(bits(ref[st]) != bits(got[st])))
        r[st] = nd
        if nd and first is None:
            first = st
    flips = {}
    for site in Q8_SITES:
        _, qa, _ = E.quantize_q8k(ref[site])
        _, qb, _ = E.quantize_q8k(got[site])
        flips[site] = int(np.sum(qa != qb))
    a, b = np.asarray(ref["l_out"], np.float64), np.asarray(got["l_out"], np.float64)
    d = np.abs(a - b)
    bad = {f"atol={t[0]:g},rtol={t[1]:g}": int(np.sum(d > t[0] + t[1] * np.abs(a))) for t in TOLS}
    return dict(first_diff=first, ndiff=r, q8_flips=flips, l_out_maxabs=float(d.max()),
                pre_ffn_maxabs=float(np.max(np.abs(np.asarray(ref["ffn_inp"], np.float64) - got["ffn_inp"]))),
                l_out_bad=bad)


def run_pair(w, toks, ops):
    lay = E.Qwen3Layer0(w, ops=ops)
    return [lay.step(E.dequant_q6k_row(w["token_embd"][t]), p) for p, t in enumerate(toks)]


def single_flip_probe(w, toks):
    """Token 1, exact pipeline; move the ffn_norm element closest to a Q8_K rounding boundary across it by the fewest ulps."""
    ref = run_pair(w, toks, {})[1]
    x = np.asarray(ref["ffn_norm"], F32)
    d, qs, _ = E.quantize_q8k(x)
    iscale = np.repeat((F32(1.0) / d).astype(F32), 256)
    v = (x * iscale).astype(F32)
    frac = np.abs(np.abs(v - np.floor(v)) - 0.5)
    frac[np.abs(x) == np.repeat(np.abs(x).reshape(-1, 256).max(1), 256)] = 9  # do not move a block max
    out = {}
    for label, idx in (("nearest-boundary element", int(np.argmin(frac))),
                       ("far-from-boundary element", int(np.argmax(np.where(frac < 9, frac, -1))))):
        y = x.copy()
        if label.startswith("far"):
            y[idx] = np.nextafter(y[idx], F32(np.inf))
        else:  # walk 1 ulp at a time, trying both directions, until exactly that code flips
            for direction in (F32(np.inf), F32(-np.inf)):
                y = x.copy()
                for _ in range(1 << 16):
                    y[idx] = np.nextafter(y[idx], direction)
                    if np.sum(E.quantize_q8k(y)[1] != qs):
                        break
                if np.sum(E.quantize_q8k(y)[1] != qs) == 1:
                    break
        _, q2, _ = E.quantize_q8k(y)
        lay = E.Qwen3Layer0(w)
        lay.step(E.dequant_q6k_row(w["token_embd"][toks[0]]), 0)
        got = lay.step(E.dequant_q6k_row(w["token_embd"][toks[1]]), 1, hooks={"ffn_norm": lambda _a, y=y: y})
        dd = np.abs(np.asarray(got["l_out"], np.float64) - np.asarray(ref["l_out"], np.float64))
        out[label] = dict(index=idx, input_change_abs=float(abs(float(y[idx]) - float(x[idx]))),
                          q8_codes_changed=int(np.sum(q2 != qs)),
                          gate_up_down_q8_flips_downstream=int(np.sum(E.quantize_q8k(got["ffn_swiglu"])[1] != E.quantize_q8k(ref["ffn_swiglu"])[1])),
                          l_out_ndiff=int(np.sum(bits(got["l_out"]) != bits(ref["l_out"]))),
                          l_out_maxabs=float(dd.max()),
                          l_out_bad_atol_1e_5=int(np.sum(dd > 1e-5)))
    return out


def main():
    out_path, models = sys.argv[1], sys.argv[2:]
    rng = np.random.default_rng(2026)
    res = {"variants": {}, "single_flip": [], "models": models}
    for m in models:
        w = E.load_weights(m, os.environ.get("LLAMA_CPP"))
        vocab = w["token_embd"].shape[0]
        pairs = [tuple(int(t) for t in rng.integers(0, vocab, 2)) for _ in range(4)]
        refs = {p: run_pair(w, p, {}) for p in pairs}
        for name, ops in VARIANTS.items():
            for p in pairs:
                got = run_pair(w, p, ops)
                rec = {"model": os.path.basename(m), "tokens": p,
                       "token0": compare(refs[p][0], got[0]), "token1": compare(refs[p][1], got[1])}
                res["variants"].setdefault(name, []).append(rec)
            v = res["variants"][name][-len(pairs):]
            print(f"{os.path.basename(m)} {name:22s} tok0 bad(1e-5)={[r['token0']['l_out_bad']['atol=1e-05,rtol=0'] for r in v]}"
                  f" tok1 bad(1e-5)={[r['token1']['l_out_bad']['atol=1e-05,rtol=0'] for r in v]}", flush=True)
        res["single_flip"].append({"model": os.path.basename(m), "tokens": pairs[0], **single_flip_probe(w, pairs[0])})
        print(json.dumps(res["single_flip"][-1]), flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
