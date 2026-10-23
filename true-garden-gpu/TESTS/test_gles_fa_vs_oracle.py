#!/usr/bin/env python3
"""Run shaders/ggml_fa_f16.comp on the current GLES driver with the Q / K-cache /
V-cache that llama.cpp itself used, and compare with llama.cpp's own
FLASH_ATTN_EXT output, bit for bit, for every decode step in a dump.

usage: test_gles_fa_vs_oracle.py [--negative-control] GLES_RUN_BINARY DUMPDIR [DUMPDIR ...]
  DUMPDIR: output of tools/dump_layer0 with FA=1 (or 2 = auto on CPU).
  --negative-control: swap the FP16 round-to-nearest-even for round-toward-zero
  (the rounding the Liverpool driver appears to use for packHalf2x16); the
  test must then FAIL on every step with n_kv >= 2 and still pass n_kv == 1.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "shaders", "ggml_exact_lib.glsl")
FA = os.path.join(HERE, "..", "shaders", "ggml_fa_f16.comp")
N_HEAD, N_HEAD_KV, D = 16, 8, 128


def manifest(d):
    rows = []
    for line in open(os.path.join(d, "MANIFEST.tsv")):
        p = line.rstrip("\n").split("\t")
        rows.append(dict(idx=int(p[0]), name=p[1], op=p[2], type=p[3], ne=[int(v) for v in p[4:8]]))
    return rows


def tensor(d, r):
    fn = [f for f in os.listdir(d) if f.startswith("%04d_" % r["idx"])][0]
    return np.fromfile(os.path.join(d, fn), dtype=np.float32 if r["type"] == "f32" else np.uint16)


def pick(rows, op, name_prefix=None):
    for r in rows:
        if r["op"] == op and (name_prefix is None or r["name"].startswith(name_prefix)):
            return r
    raise KeyError((op, name_prefix))


def main():
    args = sys.argv[1:]
    neg = "--negative-control" in args
    args = [a for a in args if a != "--negative-control"]
    gles, dumps = args[0], args[1:]
    work = os.environ.get("WORKDIR", "/tmp")
    lib = LIB
    if neg:
        src = open(LIB).read()
        rne = "((m + 0x0fffu + ((m >> 13) & 1u)) >> 13)"
        assert rne in src
        lib = os.path.join(work, "ggml_exact_lib_rtz.glsl")
        open(lib, "w").write(src.replace(rne, "(m >> 13)"))
    scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(D)))
    bad_total = steps = 0
    for dump in dumps:
        s = 0
        while os.path.isdir(os.path.join(dump, f"step{s}")):
            d = os.path.join(dump, f"step{s}")
            rows = manifest(d)
            q = tensor(d, pick(rows, "ROPE", "Qcur"))
            kc = tensor(d, pick(rows, "SET_ROWS", "cache_k")).reshape(-1, N_HEAD_KV * D)[: s + 1]
            vc = tensor(d, pick(rows, "SET_ROWS", "cache_v")).reshape(-1, N_HEAD_KV * D)[: s + 1]
            ref = tensor(d, pick(rows, "FLASH_ATTN_EXT"))
            n_kv = s + 1
            paths = {k: os.path.join(work, f"fa_{k}.bin") for k in ("p", "q", "k", "v", "o")}
            np.array([n_kv, N_HEAD, N_HEAD_KV, scale.view(np.int32)], np.int32).tofile(paths["p"])
            q.astype(np.float32).tofile(paths["q"])
            kc.astype(np.uint16).tofile(paths["k"])
            vc.astype(np.uint16).tofile(paths["v"])
            subprocess.run([gles, "--src", lib, "--src", FA, "--groups", str(N_HEAD),
                            "--in", f"0:{paths['p']}", "--in", f"1:{paths['q']}", "--in", f"2:{paths['k']}",
                            "--in", f"3:{paths['v']}", "--out", f"4:{paths['o']}:{N_HEAD * D * 4}"],
                           check=True, stderr=subprocess.DEVNULL)
            got = np.fromfile(paths["o"], np.float32)
            nd = int(np.sum(got.view(np.uint32) != ref.view(np.uint32)))
            print(f"{os.path.basename(dump)} step{s} n_kv={n_kv:3d} {'BITEXACT' if nd == 0 else 'DIFF'} "
                  f"ndiff={nd} maxabs={float(np.max(np.abs(got - ref))):.3e}")
            bad_total += nd > 0
            steps += 1
            s += 1
    print("RESULT:", "PASS" if bad_total == 0 else f"FAIL ({bad_total}/{steps} steps differ)", f"({steps} steps)")
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
