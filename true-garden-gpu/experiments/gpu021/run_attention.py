#!/usr/bin/env python3
"""Does the GPU-021 attention shader reproduce llama.cpp's FLASH_ATTN_EXT at
decode step 1 (n_kv = 2), and if not, which of its choices is responsible?

The shader (attn_two_token_gqa.comp, transcribed from the owner's read-only
preflight paste of 021-BUNDLE.tar; not hash-verified) is run unmodified except
for the GLSL ES 3.2 header and an int uniform, on the current GLES driver, with
Q / K / V from the bit-exact emulator (emu/ggml_emu.py, which equals llama.cpp
23b0202a with Jaguar flags on these models). Numpy variants of ggml's FA then
isolate its three differences from ggml: the KQ dot product summed in order
instead of ggml_vec_dot_f16's 4 x 8 lanes, a near-correctly rounded exp instead
of glibc expf, and the literal 0.08838834764831845 as the scale.

usage: LLAMA_CPP=... run_attention.py GLES_RUN OUT.json NPAIRS MODEL.gguf [MODEL.gguf ...]
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "emu"))
import ggml_emu as E  # noqa: E402

F32 = np.float32
SHADER = os.path.join(HERE, "attn_two_token_gqa.comp")


def adapted_source(path):
    src = open(path).read()
    src = src.replace("#version 310 es\n", "").replace("#extension GL_EXT_gpu_shader5 : require\n", "")
    assert "uniform uint uPosition;" in src
    return src.replace("uniform uint uPosition;", "uniform int uPositionI;\n#define uPosition uint(uPositionI)")


def run_glsl(gles, src_path, q, k16, v16, pos, work):
    cache = np.concatenate([np.concatenate([k16[p].ravel(), v16[p].ravel()]) for p in range(k16.shape[0])]).astype(F32)
    if cache.size < 4096:
        cache = np.concatenate([cache, np.zeros(4096 - cache.size, F32)])
    qp, cp, op = (os.path.join(work, n) for n in ("q.bin", "c.bin", "o.bin"))
    q.astype(F32).ravel().tofile(qp)
    cache.tofile(cp)
    subprocess.run([gles, "--src", src_path, "--groups", "16", "--in", f"0:{qp}", "--in", f"1:{cp}",
                    "--out", f"2:{op}:{16 * 128 * 4}", "--int", f"uPositionI={pos}"],
                   check=True, stderr=subprocess.DEVNULL)
    return np.fromfile(op, F32)


def dot_sequential(x16, y16):
    s = F32(0.0)
    for a, b in zip(np.asarray(x16, F32), np.asarray(y16, F32)):
        s = F32(s + F32(a * b))
    return s


def fa_variant(q, k16, v16, scale, seq_dot=False, cr_exp=False):
    saved = E.vec_dot_f16, E.expf
    if seq_dot:
        E.vec_dot_f16 = dot_sequential
    if cr_exp:
        E.expf = lambda x: F32(np.exp(np.float64(x)))
    try:
        return E.flash_attn_f16(q, k16, v16, scale).reshape(-1)
    finally:
        E.vec_dot_f16, E.expf = saved


def bits(a):
    return np.asarray(a, F32).view(np.uint32)


def main():
    gles, out_path, npairs, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    work = tempfile.mkdtemp()
    src = os.path.join(work, "attn021.comp")
    open(src, "w").write(adapted_source(SHADER))
    rng = np.random.default_rng(2026)
    scale021 = F32(0.08838834764831845)
    res = {"cases": [], "scale_ggml": None, "scale_021": float(scale021)}
    for m in models:
        w = E.load_weights(m, os.environ.get("LLAMA_CPP"))
        vocab = w["token_embd"].shape[0]
        for _ in range(npairs):
            toks = [int(t) for t in rng.integers(0, vocab, 2)]
            emb = [E.dequant_q6k_row(w["token_embd"][t]) for t in toks]
            lay = E.Qwen3Layer0(w)
            t0 = lay.step(emb[0], 0)
            k16_0, v16_0 = np.stack(lay.k16), np.stack(lay.v16)
            t1 = lay.step(emb[1], 1)
            k16, v16 = np.stack(lay.k16), np.stack(lay.v16)
            res["scale_ggml"] = float(lay.kq_scale)
            q1 = t1["Qrope"]
            ref = t1["kqv_out"]
            g0 = run_glsl(gles, src, t0["Qrope"], k16_0, v16_0, 0, work)
            g1 = run_glsl(gles, src, q1, k16, v16, 1, work)
            c = {"model": os.path.basename(m), "tokens": toks,
                 "glsl021_step0_ndiff": int(np.sum(bits(g0) != bits(t0["kqv_out"]))),
                 "glsl021_step1_ndiff": int(np.sum(bits(g1) != bits(ref)))}
            for name, kw, sc in (("np021_all", dict(seq_dot=True, cr_exp=True), scale021),
                                 ("seq_dot_only", dict(seq_dot=True), lay.kq_scale),
                                 ("cr_exp_only", dict(cr_exp=True), lay.kq_scale),
                                 ("scale021_only", {}, scale021)):
                c[name + "_ndiff"] = int(np.sum(bits(fa_variant(q1, k16, v16, sc, **kw)) != bits(ref)))
            # what the 021 attention output alone does to the rest of the layer
            lay2 = E.Qwen3Layer0(w)
            lay2.step(emb[0], 0)
            t1x = lay2.step(emb[1], 1, hooks={"kqv_out": lambda o, g=g1: g.copy()})
            d = np.abs(t1x["l_out"].astype(np.float64) - t1["l_out"].astype(np.float64))
            c["l_out_ndiff"] = int(np.sum(bits(t1x["l_out"]) != bits(t1["l_out"])))
            c["l_out_bad_1e-5"] = int(np.sum(d > 1e-5))
            c["l_out_maxabs"] = float(d.max())
            res["cases"].append(c)
            print(json.dumps(c), flush=True)
    cs = res["cases"]
    n = len(cs)
    summ = {k: sum(1 for c in cs if c[k] > 0) for k in cs[0] if k.endswith("ndiff")}
    res["summary"] = {"cases": n, "cases_with_any_diff": summ,
                      "median_l_out_bad_1e-5_when_attn_differs":
                          float(np.median([c["l_out_bad_1e-5"] for c in cs if c["glsl021_step1_ndiff"]] or [0]))}
    print(json.dumps(res["summary"]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
