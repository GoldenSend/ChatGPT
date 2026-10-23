#!/usr/bin/env python3
"""End-to-end ggml-exact Qwen3 layer 0 on the GLES driver, compared stage by
stage with llama.cpp's own tensors for every decode step of a dump.

Only the layer input row (the token embedding, `embd`) and the RoPE cos/sin
table are taken from the host; every other stage, including the FP16 K/V
cache, is produced by the shaders from the previous shader output. So a PASS
means the GPU chain reproduces llama.cpp bit for bit, not just each op in
isolation.

usage: test_gles_layer_vs_oracle.py GLES_RUN MODEL.gguf DUMPDIR
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "emu"))
import ggml_emu as E  # noqa: E402

SH = os.path.join(HERE, "..", "shaders")
LIB = os.path.join(SH, "ggml_exact_lib.glsl")
N_EMBD, N_HEAD, N_HEAD_KV, HD, EPS = 1024, 16, 8, 128, np.float32(1e-6)

STAGES = [  # (our name, oracle name, oracle op)
    ("attn_norm", "attn_norm-0", "MUL"), ("Qcur", "Qcur-0", "MUL_MAT"), ("Kcur", "Kcur-0", "MUL_MAT"),
    ("Vcur", "Vcur-0", "MUL_MAT"), ("Qcur_normed", "Qcur_normed-0", "MUL"), ("Kcur_normed", "Kcur_normed-0", "MUL"),
    ("Qrope", "Qcur-0", "ROPE"), ("Krope", "Kcur-0", "ROPE"), ("kqv_out", None, "FLASH_ATTN_EXT"),
    ("attn_out", None, "MUL_MAT@wo"), ("ffn_inp", "ffn_inp-0", "ADD"), ("ffn_norm", "ffn_norm-0", "MUL"),
    ("ffn_gate", "ffn_gate-0", "MUL_MAT"), ("ffn_up", "ffn_up-0", "MUL_MAT"), ("ffn_swiglu", "ffn_swiglu-0", "SWIGLU"),
    ("ffn_out", "ffn_out-0", "MUL_MAT"), ("l_out", "l_out-0", "ADD"),
]


class Gpu:
    def __init__(self, gles, work):
        self.gles, self.work, self.n = gles, work, 0

    def f(self, tag):
        self.n += 1
        return os.path.join(self.work, f"L{self.n:03d}_{tag}.bin")

    def put(self, arr, tag):
        p = self.f(tag)
        np.ascontiguousarray(arr).tofile(p)
        return p

    def run(self, shader, groups, ins, out_binding, out_bytes, ints=()):
        out = self.f("out")
        cmd = [self.gles, "--src", LIB, "--src", os.path.join(SH, shader), "--groups", str(groups)]
        for b, p in ins:
            cmd += ["--in", f"{b}:{p}"]
        cmd += ["--out", f"{out_binding}:{out}:{out_bytes}"]
        for k, v in ints:
            cmd += ["--int", f"{k}={v}"]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return out

    def rms_norm(self, x_path, rows, n, w):
        prm = self.put(np.array([n, rows, EPS.view(np.int32), 1], np.int32), "p")
        return self.run("ggml_rms_norm_mul.comp", rows, [(0, prm), (1, x_path), (2, self.put(w, "w"))], 3, rows * n * 4)

    def matvec(self, raw, x_path, qtype):
        nrow, nb = raw.shape[:2]
        q8 = self.run("ggml_q8k_quantize.comp", nb, [(0, x_path)], 1, nb * 273 * 4)
        prm = self.put(np.array([nrow, nb, qtype], np.int32), "p")
        return self.run("ggml_matvec_qk.comp", (nrow + 63) // 64, [(0, prm), (1, self.put(raw, "w")), (2, q8)], 3, nrow * 4)

    def rope(self, x_path, heads, tab):
        prm = self.put(np.array([heads, HD], np.int32), "p")
        return self.run("ggml_rope_neox.comp", heads, [(0, prm), (1, x_path), (2, self.put(tab, "tab"))], 3, heads * HD * 4)

    def elt(self, mode, a_path, b_path, n):
        nout = n // 2 if mode == 2 else n
        return self.run("ggml_eltwise.comp", (n + 63) // 64, [(0, a_path), (1, b_path)], 2, nout * 4,
                        [("mode", mode), ("n", n)])

    def fa(self, q_path, k_bits, v_bits, n_kv):
        scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(HD)))
        prm = self.put(np.array([n_kv, N_HEAD, N_HEAD_KV, scale.view(np.int32)], np.int32), "p")
        return self.run("ggml_fa_f16.comp", N_HEAD, [(0, prm), (1, q_path), (2, self.put(k_bits, "k")),
                                                      (3, self.put(v_bits, "v"))], 4, N_HEAD * HD * 4)


def manifest(d):
    rows = []
    for line in open(os.path.join(d, "MANIFEST.tsv")):
        p = line.rstrip("\n").split("\t")
        rows.append(dict(idx=int(p[0]), name=p[1], op=p[2], type=p[3], srcs=p[8:]))
    return rows


def oracle(d, rows, name, op):
    if op == "MUL_MAT@wo":
        r = next(r for r in rows if r["op"] == "MUL_MAT" and any(s.startswith("kqv_out") for s in r["srcs"]))
    else:
        r = next(r for r in rows if r["op"] == op and (name is None or r["name"] == name))
    fn = [f for f in os.listdir(d) if f.startswith("%04d_" % r["idx"])][0]
    return np.fromfile(os.path.join(d, fn), np.float32)


def main():
    gles, model, dump = sys.argv[1:4]
    work = os.path.join(os.environ.get("WORKDIR", "/tmp"), "layer")
    os.makedirs(work, exist_ok=True)
    w = E.load_weights(model, os.environ.get("LLAMA_CPP"))
    g = Gpu(gles, work)
    rd = lambda p: np.fromfile(p, np.float32)  # noqa: E731
    qt = lambda name: 4 if w[name + ".type"] == "Q4_K" else 6  # noqa: E731
    k_cache, v_cache = [], []
    bad = total = 0
    s = 0
    while os.path.isdir(os.path.join(dump, f"step{s}")):
        d = os.path.join(dump, f"step{s}")
        rows = manifest(d)
        out = {}
        x = g.put(oracle(d, rows, "embd", "GET_ROWS"), "embd")
        out["attn_norm"] = a = g.rms_norm(x, 1, N_EMBD, w["attn_norm"])
        out["Qcur"] = q = g.matvec(w["attn_q"], a, qt("attn_q"))
        out["Kcur"] = k = g.matvec(w["attn_k"], a, qt("attn_k"))
        out["Vcur"] = v = g.matvec(w["attn_v"], a, qt("attn_v"))
        out["Qcur_normed"] = qn = g.rms_norm(q, N_HEAD, HD, w["attn_q_norm"])
        out["Kcur_normed"] = kn = g.rms_norm(k, N_HEAD_KV, HD, w["attn_k_norm"])
        cos, sin = E.rope_neox_cache(s, HD, 1e6)  # host table, same code path as ggml
        tab = np.concatenate([cos, sin]).astype(np.float32)
        out["Qrope"] = qr = g.rope(qn, N_HEAD, tab)
        out["Krope"] = kr = g.rope(kn, N_HEAD_KV, tab)
        k_cache.append(np.fromfile(g.elt(2, kr, kr, N_HEAD_KV * HD), np.uint32))
        v_cache.append(np.fromfile(g.elt(2, v, v, N_HEAD_KV * HD), np.uint32))
        out["kqv_out"] = o = g.fa(qr, np.concatenate(k_cache), np.concatenate(v_cache), s + 1)
        out["attn_out"] = ao = g.matvec(w["attn_output"], o, qt("attn_output"))
        out["ffn_inp"] = fi = g.elt(0, ao, x, N_EMBD)
        out["ffn_norm"] = fn = g.rms_norm(fi, 1, N_EMBD, w["ffn_norm"])
        out["ffn_gate"] = gt = g.matvec(w["ffn_gate"], fn, qt("ffn_gate"))
        out["ffn_up"] = up = g.matvec(w["ffn_up"], fn, qt("ffn_up"))
        out["ffn_swiglu"] = sw = g.elt(1, gt, up, 3072)
        out["ffn_out"] = fo = g.matvec(w["ffn_down"], sw, qt("ffn_down"))
        out["l_out"] = g.elt(0, fo, fi, N_EMBD)
        first = None
        for ours, oname, oop in STAGES:
            ref = oracle(d, rows, oname, oop)
            got = rd(out[ours])
            nd = int(np.sum(got.view(np.uint32) != ref.view(np.uint32)))
            if nd and first is None:
                first = f"{ours} ndiff={nd} maxabs={float(np.max(np.abs(got - ref))):.3e}"
        lo_ref, lo = oracle(d, rows, "l_out-0", "ADD"), rd(out["l_out"])
        nd = int(np.sum(lo.view(np.uint32) != lo_ref.view(np.uint32)))
        print(f"step{s} n_kv={s + 1:3d} l_out {'BITEXACT' if nd == 0 else 'DIFF'} ndiff={nd} "
              f"first differing stage: {first or 'none'}")
        bad += first is not None
        total += 1
        s += 1
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad}/{total} steps have a differing stage)", f"({total} steps x 17 stages)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
