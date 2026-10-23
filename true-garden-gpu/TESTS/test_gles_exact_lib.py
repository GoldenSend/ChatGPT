#!/usr/bin/env python3
"""Run the GLSL helpers (shaders/ggml_exact_lib.glsl) on the current GLES driver and
compare with the CPU reference: FP16 RNE rounding, glibc expf, correctly rounded
1/x and sqrt, ggml_v_expf, SiLU, and ggml's RMSNorm on stress rows with outliers.
On a new driver (radeonsi) this is the first test to run: it shows whether the
driver keeps IEEE fp32 behaviour (denormals, no contraction under `precise`).

usage: test_gles_exact_lib.py GLES_RUN_BINARY WORKDIR [N]
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "emu"))
import ggml_emu as E  # noqa: E402
import glibc_expf as G  # noqa: E402

LIB = os.path.join(HERE, "..", "shaders", "ggml_exact_lib.glsl")
UNIT = os.path.join(HERE, "..", "shaders", "unit_exact_lib.comp")


def run(gles, work, mode, x):
    xi, yo = os.path.join(work, "u_in.bin"), os.path.join(work, "u_out.bin")
    x.astype(np.float32).tofile(xi)
    groups = min(4096, (x.size + 63) // 64)
    subprocess.run([gles, "--src", LIB, "--src", UNIT, "--groups", str(groups), "--in", f"0:{xi}",
                    "--out", f"1:{yo}:{x.size * 4}", "--int", f"mode={mode}", "--int", f"n={x.size}"],
                   check=True, stderr=subprocess.DEVNULL)
    return np.fromfile(yo, np.float32)


def report(name, x, got, ref):
    gb, rb = got.view(np.uint32), ref.view(np.uint32)
    nan_both = np.isnan(got) & np.isnan(ref)
    bad = (gb != rb) & ~nan_both
    print(f"{name:34s} n={x.size:9d} mismatches={int(bad.sum())}")
    if bad.any():
        i = np.flatnonzero(bad)[:5]
        print("   e.g.", [(float(x[j]), float(got[j]), float(ref[j])) for j in i])
    return int(bad.sum())


def main():
    gles, work = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 22
    rng = np.random.default_rng(11)
    total = 0

    # FP16 rounding: random bit patterns over the whole f32 range plus the f16 subnormal / overflow edges
    bits = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    edge = np.concatenate([np.arange(0x33000000 - 64, 0x38800000 + 64, 997, dtype=np.uint32),
                           np.arange(0x477fe000 - 4096, 0x47800000 + 64, 1, dtype=np.uint32)])
    x = np.concatenate([bits, edge, edge | np.uint32(0x80000000)]).view(np.float32)
    with np.errstate(over="ignore"):
        total += report("f16r (f32->f16 RNE->f32)", x, run(gles, work, 0, x), x.astype(np.float16).astype(np.float32))

    # glibc expf: attention arguments are s - M <= 0; cover [-87.3, 0] densely plus small positives
    x = np.concatenate([-rng.random(n) * 87.3, (rng.random(n // 8) - 0.5) * 8, np.array([0.0, -0.0])]).astype(np.float32)
    got = run(gles, work, 1, x)
    ref = G.expf_generic_vec(x)
    cr = np.exp(x.astype(np.float64)).astype(np.float32)
    miss = got.view(np.uint32) != ref.view(np.uint32)
    # Known limitation: the fp32-only port carries ~48 bits, so when glibc's own
    # double-precision result lands within ~2^-47 of a float midpoint (glibc then
    # misrounds) the port returns the correctly rounded value instead.
    known = miss & (got == cr) & (ref != cr)
    print(f"{'glibc_expf on [-87.3, 4]':34s} n={x.size:9d} mismatches={int(miss.sum())} "
          f"(known class: glibc not correctly rounded, port is: {int(known.sum())})")
    if (miss & ~known).any():
        i = np.flatnonzero(miss & ~known)[:5]
        print("   unexplained e.g.", [(float(x[j]), float(got[j]), float(ref[j])) for j in i])
    total += int((miss & ~known).sum())

    # correctly rounded reciprocal on the range S can take (1 <= S <= n_kv)
    x = np.concatenate([1 + rng.random(n) * 4095, rng.random(n // 8) * 4 + 1]).astype(np.float32)
    total += report("rcp_cr on [1, 4096]", x, run(gles, work, 2, x), (np.float32(1) / x).astype(np.float32))

    # ggml_v_expf (SSE2, no FMA) over its whole useful range, including the |n| > 126 branches
    x = np.concatenate([(rng.random(n) - 0.5) * 200, (rng.random(n // 8) - 0.5) * 20]).astype(np.float32)
    total += report("ggml_v_expf on [-100, 100]", x, run(gles, work, 3, x), E.v_expf(x))

    # correctly rounded sqrt over mean-square values RMSNorm can see
    x = np.exp(rng.uniform(np.log(1e-12), np.log(1e8), n)).astype(np.float32)
    total += report("sqrt_cr on [1e-12, 1e8]", x, run(gles, work, 4, x), np.sqrt(x).astype(np.float32))

    # ggml silu (swiglu with up = 1) on gate values
    x = np.concatenate([(rng.random(n) - 0.5) * 40, rng.standard_normal(n // 4) * 2]).astype(np.float32)
    total += report("ggml silu on [-20, 20]", x, run(gles, work, 5, x), E.swiglu(x, np.ones_like(x)))

    # RMSNorm kernel: double-float sequential sum vs ggml's double sum, including
    # Qwen-style massive activations (a few channels in the thousands)
    rows = 4096
    X = (rng.standard_normal((rows, 1024)) * np.exp(rng.uniform(-6, 3, (rows, 1)))).astype(np.float32)
    X[::7, rng.integers(0, 1024, rows // 7 + 1)[: len(X[::7])]] *= 3000
    Wn = (1 + rng.standard_normal(1024) * 0.3).astype(np.float32)
    prm, xi, wi, yo = (os.path.join(work, f"rn_{k}.bin") for k in "pxwy")
    np.array([1024, rows, np.float32(1e-6).view(np.int32), 1], np.int32).tofile(prm)
    X.tofile(xi)
    Wn.tofile(wi)
    subprocess.run([gles, "--src", LIB, "--src", os.path.join(HERE, "..", "shaders", "ggml_rms_norm_mul.comp"),
                    "--groups", str(rows), "--in", f"0:{prm}", "--in", f"1:{xi}", "--in", f"2:{wi}",
                    "--out", f"3:{yo}:{rows * 1024 * 4}"], check=True, stderr=subprocess.DEVNULL)
    got = np.fromfile(yo, np.float32)
    ref = (E.rms_norm(X, 1e-6) * Wn).astype(np.float32).reshape(-1)
    total += report(f"rms_norm*w ({rows} rows x 1024)", got, got, ref)

    print("RESULT:", "PASS" if total == 0 else f"FAIL ({total} unexplained mismatches)",
          "(expf known-class mismatches are listed above and are not counted)")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
