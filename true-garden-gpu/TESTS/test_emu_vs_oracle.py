#!/usr/bin/env python3
"""Compare the numpy emulator against tensors dumped from llama.cpp itself.

usage: test_emu_vs_oracle.py MODEL.gguf DUMPDIR TOKEN0 TOKEN1 [...]
  DUMPDIR is the output of tools/dump_layer0 (run with FA=1).
Exit 0 only if every compared tensor is bit-identical.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "emu"))
import ggml_emu as E  # noqa: E402

# (emulator name, oracle tensor name, oracle op); first match in the manifest wins
PAIRS = [
    ("inp", "embd", "GET_ROWS"),
    ("attn_norm", "attn_norm-0", "MUL"),
    ("Qcur", "Qcur-0", "MUL_MAT"),
    ("Kcur", "Kcur-0", "MUL_MAT"),
    ("Vcur", "Vcur-0", "MUL_MAT"),
    ("Qcur_normed", "Qcur_normed-0", "MUL"),
    ("Kcur_normed", "Kcur_normed-0", "MUL"),
    ("Qrope", "Qcur-0", "ROPE"),
    ("Krope", "Kcur-0", "ROPE"),
    ("kqv_out", None, "FLASH_ATTN_EXT"),
    ("attn_out", None, "MUL_MAT@wo"),
    ("ffn_inp", "ffn_inp-0", "ADD"),
    ("ffn_norm", "ffn_norm-0", "MUL"),
    ("ffn_gate", "ffn_gate-0", "MUL_MAT"),
    ("ffn_up", "ffn_up-0", "MUL_MAT"),
    ("ffn_swiglu", "ffn_swiglu-0", "SWIGLU"),
    ("ffn_out", "ffn_out-0", "MUL_MAT"),
    ("l_out", "l_out-0", "ADD"),
]


def load_step(d):
    rows = []
    with open(os.path.join(d, "MANIFEST.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append(dict(idx=int(p[0]), name=p[1], op=p[2], type=p[3], srcs=p[8:]))
    return rows


def find(rows, name, op):
    if op == "MUL_MAT@wo":  # output projection: MUL_MAT whose src1 is kqv_out
        for r in rows:
            if r["op"] == "MUL_MAT" and any(s.startswith("kqv_out") for s in r["srcs"]):
                return r
        return None
    for r in rows:
        if r["op"] == op and (name is None or r["name"] == name):
            return r
    return None


def read(d, r):
    fn = [f for f in os.listdir(d) if f.startswith("%04d_" % r["idx"])][0]
    return np.fromfile(os.path.join(d, fn), dtype=np.float32 if r["type"] == "f32" else np.float16)


def main():
    model, dump = sys.argv[1], sys.argv[2]
    toks = [int(t) for t in sys.argv[3:]]
    w = E.load_weights(model, os.environ.get("LLAMA_CPP"))
    layer = E.Qwen3Layer0(w)
    bad = 0
    for s, tok in enumerate(toks):
        d = os.path.join(dump, "step%d" % s)
        rows = load_step(d)
        x = E.dequant_q6k_row(w["token_embd"][tok])
        t = layer.step(x, s)
        for en, on, op in PAIRS:
            r = find(rows, on, op)
            if r is None:
                print(f"step{s} {en:12s} MISSING in oracle")
                bad += 1
                continue
            ref = read(d, r).astype(np.float32)
            got = np.asarray(t[en], np.float32).reshape(-1)
            same = np.array_equal(ref.view(np.uint32), got.view(np.uint32))
            ndiff = int(np.sum(ref.view(np.uint32) != got.view(np.uint32)))
            maxd = float(np.max(np.abs(ref - got))) if ref.shape == got.shape else float("nan")
            print(f"step{s} {en:12s} {'BITEXACT' if same else 'DIFF':8s} n={ref.size:5d} ndiff={ndiff:5d} maxabs={maxd:.3e}")
            bad += 0 if same else 1
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad} tensors differ)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
