#!/usr/bin/env python3
"""Run shaders/ggml_q8k_quantize.comp + ggml_matvec_qk.comp on the current GLES
driver for every layer-0 projection in a llama.cpp dump and compare with
llama.cpp's own MUL_MAT outputs, bit for bit. Also checks the GPU Q8_K blocks
against the emulator's quantize_q8k.

usage: test_gles_matvec_vs_oracle.py [--negative-control] GLES_RUN MODEL.gguf DUMPDIR
  --negative-control: replace the 8-lane float accumulation with a single
  row-serial accumulator (what a straightforward GPU kernel does); some rows
  must then differ.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "emu"))
import ggml_emu as E  # noqa: E402

LIB = os.path.join(HERE, "..", "shaders", "ggml_exact_lib.glsl")
QZ = os.path.join(HERE, "..", "shaders", "ggml_q8k_quantize.comp")
MV = os.path.join(HERE, "..", "shaders", "ggml_matvec_qk.comp")

# (weight, input tensor (name, op), output tensor (name, op))
PROJ = [
    ("attn_q", ("attn_norm-0", "MUL"), ("Qcur-0", "MUL_MAT")),
    ("attn_k", ("attn_norm-0", "MUL"), ("Kcur-0", "MUL_MAT")),
    ("attn_v", ("attn_norm-0", "MUL"), ("Vcur-0", "MUL_MAT")),
    ("attn_output", (None, "FLASH_ATTN_EXT"), (None, "MUL_MAT@wo")),
    ("ffn_gate", ("ffn_norm-0", "MUL"), ("ffn_gate-0", "MUL_MAT")),
    ("ffn_up", ("ffn_norm-0", "MUL"), ("ffn_up-0", "MUL_MAT")),
    ("ffn_down", ("ffn_swiglu-0", "SWIGLU"), ("ffn_out-0", "MUL_MAT")),
]


def manifest(d):
    rows = []
    for line in open(os.path.join(d, "MANIFEST.tsv")):
        p = line.rstrip("\n").split("\t")
        rows.append(dict(idx=int(p[0]), name=p[1], op=p[2], type=p[3], srcs=p[8:]))
    return rows


def find(rows, name, op):
    if op == "MUL_MAT@wo":
        return next(r for r in rows if r["op"] == "MUL_MAT" and any(s.startswith("kqv_out") for s in r["srcs"]))
    return next(r for r in rows if r["op"] == op and (name is None or r["name"] == name))


def tensor(d, r):
    fn = [f for f in os.listdir(d) if f.startswith("%04d_" % r["idx"])][0]
    return np.fromfile(os.path.join(d, fn), dtype=np.float32)


def main():
    args = sys.argv[1:]
    neg = "--negative-control" in args
    args = [a for a in args if a != "--negative-control"]
    gles, model, dump = args
    work = os.environ.get("WORKDIR", "/tmp")
    mv = MV
    if neg:
        src = open(MV).read()
        assert src.count("acc[l] = t + acc[l];") == 2
        src = src.replace("acc[l] = t + acc[l];", "acc[0] = t + acc[0];")
        mv = os.path.join(work, "ggml_matvec_qk_serial.comp")
        open(mv, "w").write(src)
    w = E.load_weights(model, os.environ.get("LLAMA_CPP"))
    p = {k: os.path.join(work, f"mv_{k}.bin") for k in ("x", "q8", "w", "p", "y")}
    bad = total = q8bad = 0
    s = 0
    while os.path.isdir(os.path.join(dump, f"step{s}")):
        d = os.path.join(dump, f"step{s}")
        rows = manifest(d)
        for wname, (iname, iop), (oname, oop) in PROJ:
            x = tensor(d, find(rows, iname, iop))
            ref = tensor(d, find(rows, oname, oop))
            raw = w[wname]
            nrow, nb = raw.shape[0], raw.shape[1]
            x.tofile(p["x"])
            subprocess.run([gles, "--src", LIB, "--src", QZ, "--groups", str(nb), "--in", f"0:{p['x']}",
                            "--out", f"1:{p['q8']}:{nb * 273 * 4}"], check=True, stderr=subprocess.DEVNULL)
            q8 = np.fromfile(p["q8"], np.uint32).reshape(nb, 273)
            ed, eq, eb = E.quantize_q8k(x)
            q8ok = (np.array_equal(q8[:, 0].view(np.float32), ed) and np.array_equal(q8[:, 1:257].view(np.int32), eq)
                    and np.array_equal(q8[:, 257:].view(np.int32), eb))
            q8bad += not q8ok
            raw.tofile(p["w"])
            qtype = 4 if w[wname + ".type"] == "Q4_K" else 6
            np.array([nrow, nb, qtype], np.int32).tofile(p["p"])
            subprocess.run([gles, "--src", LIB, "--src", mv, "--groups", str((nrow + 63) // 64),
                            "--in", f"0:{p['p']}", "--in", f"1:{p['w']}", "--in", f"2:{p['q8']}",
                            "--out", f"3:{p['y']}:{nrow * 4}"], check=True, stderr=subprocess.DEVNULL)
            got = np.fromfile(p["y"], np.float32)
            nd = int(np.sum(got.view(np.uint32) != ref.view(np.uint32)))
            print(f"step{s} {wname:12s} {w[wname + '.type']} rows={nrow:5d} q8={'OK' if q8ok else 'DIFF'} "
                  f"{'BITEXACT' if nd == 0 else 'DIFF'} ndiff={nd} maxabs={float(np.max(np.abs(got - ref))):.3e}")
            bad += nd > 0
            total += 1
        s += 1
    print("RESULT:", "PASS" if bad == 0 and q8bad == 0 else f"FAIL ({bad}/{total} matvecs differ, {q8bad} Q8_K blocks sets differ)")
    return 0 if bad == 0 and q8bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
