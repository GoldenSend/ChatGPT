#!/usr/bin/env python3
"""Model of fp32 denormal flushing (FTZ/DAZ), as GCN radeonsi does for GLSL compute.

1. Predicts the GPU-022 part B output of ggml_v_expf under FTZ, so it can be
   compared bit for bit with the Liverpool output (sha256 printed).
2. Checks whether FTZ inside ggml_v_expf (SiLU) and expf (flash attention) can
   change any layer-0 stage: random 7-step sequences on the synthetic models,
   every stage compared bit for bit with and without FTZ.

usage: LLAMA_CPP=... ftz_model.py BUNDLE_DIR NSEQ MODEL.gguf [MODEL.gguf ...]
"""
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "emu"))
import ggml_emu as E  # noqa: E402

F32 = np.float32
TINY = np.finfo(F32).tiny
H = float.fromhex


def ftz(a):
    a = np.asarray(a, F32)
    return np.where((a != 0) & (np.abs(a) < TINY), np.copysign(F32(0), a), a).astype(F32)


def v_expf_ftz(x):
    """ggml_v_expf exactly as shaders/ggml_exact_lib.glsl, every fp32 result flushed."""
    x = ftz(x)
    with np.errstate(over="ignore", invalid="ignore"):
        r = F32(12582912.0)
        z = ftz(ftz(x * F32(H('0x1.715476p+0'))) + r)
        n = ftz(z - r)
        b = ftz(ftz(x - ftz(n * F32(H('0x1.62e4p-1')))) - ftz(n * F32(H('0x1.7f7d1cp-20'))))
        e = (z.view(np.uint32) << np.uint32(23)).astype(np.uint32)
        k = ftz((e + np.array(1.0, F32).view(np.uint32)).astype(np.uint32).view(F32))
        c = np.abs(n) > F32(126)
        u = ftz(b * b)
        t1 = ftz(ftz(F32(H('0x1.0e4020p-7')) * b) + F32(H('0x1.573e2ep-5')))
        t2 = ftz(ftz(F32(H('0x1.555e66p-3')) * b) + F32(H('0x1.fffdb6p-2')))
        t3 = ftz(ftz(t1 * u) + t2)
        j = ftz(ftz(t3 * u) + ftz(F32(H('0x1.ffffecp-1')) * b))
        res = ftz(ftz(j * k) + k)
        g = np.where(n <= 0, np.uint32(0x82000000), np.uint32(0)).astype(np.uint32)
        s1 = ftz((g + np.uint32(0x7f000000)).astype(np.uint32).view(F32))
        s2 = ftz((e - g).astype(np.uint32).view(F32))
        alt = np.where(np.abs(n) > F32(192), ftz(s1 * s1), ftz(ftz(ftz(s2 * j) + s2) * s1))
        return np.where(c, alt, res).astype(F32)


def part_b(bundle):
    line = [ln for ln in open(os.path.join(bundle, "plan", "plan_B.sh")) if ln.startswith("run B4_")][0]
    x = np.fromfile(os.path.join(bundle, "in", re.search(r'0:\$I/([0-9a-f]+\.bin)', line).group(1)), F32)
    exp = np.fromfile(os.path.join(bundle, "expected", "B4_ggml_v_expf.bin"), F32)
    got = v_expf_ftz(x)
    d = got.view(np.uint32) != exp.view(np.uint32)
    print(f"B4 under FTZ: {int(d.sum())} of {x.size} values differ from expected, first at index {int(np.argmax(d))}, "
          f"x in [{float(x[d].min()):.4f}, {float(x[d].max()):.4f}]; "
          f"sha256 of predicted output {hashlib.sha256(got.tobytes()).hexdigest()}")


def layer_check(models, nseq):
    rng = np.random.default_rng(7)
    real_v, real_e = E.v_expf, E.expf
    n = bad = 0
    for m in models:
        w = E.load_weights(m, os.environ.get("LLAMA_CPP"))
        vocab = w["token_embd"].shape[0]
        for _ in range(nseq):
            toks = [int(t) for t in rng.integers(0, vocab, 7)]
            outs = []
            for flush in (False, True):
                E.v_expf = v_expf_ftz if flush else real_v
                E.expf = (lambda x: ftz(real_e(x))[()]) if flush else real_e
                lay = E.Qwen3Layer0(w)
                outs.append([lay.step(E.dequant_q6k_row(w["token_embd"][t]), p) for p, t in enumerate(toks)])
            E.v_expf, E.expf = real_v, real_e
            for a, b in zip(*outs):
                for st in a:
                    n += 1
                    if not np.array_equal(np.asarray(a[st], F32).view(np.uint32), np.asarray(b[st], F32).view(np.uint32)):
                        bad += 1
    print(f"layer 0 with FTZ in ggml_v_expf and expf: {n - bad} of {n} stage outputs bit-identical "
          f"({len(models)} models x {nseq} sequences x 7 steps)")


if __name__ == "__main__":
    part_b(sys.argv[1])
    layer_check(sys.argv[3:], int(sys.argv[2]))
