#!/usr/bin/env python3
"""Build the GPU-022 bundle: a fixed list of dispatches (a "plan") that the PS4
replays with nothing but bash, python3 and tools/gles_run.py, plus every host
input it needs and the reference each GPU output must equal.

  Part B: the GLSL helpers (FP16 RNE, expf, rcp, ggml_v_expf, sqrt, SiLU,
          RMSNorm) on fixed inputs, as in TESTS/test_gles_exact_lib.py.
  Part C: layer 0 chained on the GPU for NSTEPS decode steps, as in
          TESTS/test_gles_layer_vs_oracle.py. Host inputs are only the weights,
          the embedding row, the RoPE table and small parameter blocks; the K/V
          cache is concatenated from the GPU's own earlier outputs.

usage: LLAMA_CPP=... make_gate_plan.py MODEL.gguf DUMPDIR NSTEPS OUTDIR
writes OUTDIR/{in/, refs/, plan/plan_B.sh, plan/plan_C.sh, plan/labels.tsv}.
refs/ holds the oracle (llama.cpp tensors for C, CPU references for B); it is
used offline to verify a replay, and the verified replay outputs become
expected/ in the bundle (see tools/build_gate_bundle.sh).
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "emu"))
sys.path.insert(0, os.path.join(HERE, "..", "TESTS"))
import ggml_emu as E  # noqa: E402
import glibc_expf as G  # noqa: E402
from test_gles_layer_vs_oracle import STAGES, manifest, oracle  # noqa: E402

N_EMBD, N_HEAD, N_HEAD_KV, HD, EPS = 1024, 16, 8, 128, np.float32(1e-6)


class Plan:
    def __init__(self, out):
        self.out = out
        for d in ("in", "refs", "plan"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        self.lines = {"B": [], "C": []}
        self.labels = []

    def put(self, arr):
        b = np.ascontiguousarray(arr).tobytes()
        name = hashlib.sha256(b).hexdigest()[:24] + ".bin"
        p = os.path.join(self.out, "in", name)
        if not os.path.exists(p):
            open(p, "wb").write(b)
        return "$I/" + name

    def run(self, part, label, shader, groups, ins, out_binding, out_bytes, ints=(), step=-1, stage="-", ref=None):
        args = [f"--groups {groups}"] + [f'--in "{b}:{p}"' for b, p in ins]
        args.append(f'--out "{out_binding}:$O/{label}.bin:{out_bytes}"')
        args += [f"--int {k}={v}" for k, v in ints]
        self.lines[part].append(f"run {label} {shader} " + " ".join(args))
        has_ref = ref is not None
        if has_ref:
            np.ascontiguousarray(ref).tofile(os.path.join(self.out, "refs", label + ".bin"))
        self.labels.append((label, part, step, stage, "ref" if has_ref else "chain"))
        return "$O/" + label + ".bin"

    def write(self):
        for part, lines in self.lines.items():
            open(os.path.join(self.out, "plan", f"plan_{part}.sh"), "w").write(
                f"# GPU-022 part {part}: {sum(l.startswith('run ') for l in lines)} dispatches. Sourced by tools/gpu022_gate.sh.\n"
                + "\n".join(lines) + "\n")
        with open(os.path.join(self.out, "plan", "labels.tsv"), "w") as f:
            for row in self.labels:
                f.write("\t".join(map(str, row)) + "\n")


def part_b(P, n=65536, rows=256):
    """Same inputs and references as TESTS/test_gles_exact_lib.py (seed 11), smaller RMSNorm."""
    rng = np.random.default_rng(11)

    def unit(label, mode, x, ref):
        x = x.astype(np.float32)
        P.run("B", label, "unit_exact_lib.comp", min(4096, (x.size + 63) // 64), [(0, P.put(x))], 1, x.size * 4,
              [("mode", mode), ("n", x.size)], stage=label, ref=ref.astype(np.float32))

    bits = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    edge = np.concatenate([np.arange(0x33000000 - 64, 0x38800000 + 64, 997, dtype=np.uint32),
                           np.arange(0x477fe000 - 4096, 0x47800000 + 64, 1, dtype=np.uint32)])
    x = np.concatenate([bits, edge, edge | np.uint32(0x80000000)]).view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        unit("B1_f16r", 0, x, x.astype(np.float16).astype(np.float32))
    x = np.concatenate([-rng.random(n) * 87.3, (rng.random(n // 8) - 0.5) * 8, np.array([0.0, -0.0])]).astype(np.float32)
    ref, cr = G.expf_generic_vec(x), np.exp(x.astype(np.float64)).astype(np.float32)
    unit("B2_glibc_expf", 1, x, ref)
    # known class (see the test): where glibc is not correctly rounded, the port may return the correctly rounded value
    cr.astype(np.float32).tofile(os.path.join(P.out, "refs", "B2_glibc_expf.alt.bin"))
    x = np.concatenate([1 + rng.random(n) * 4095, rng.random(n // 8) * 4 + 1]).astype(np.float32)
    unit("B3_rcp_cr", 2, x, np.float32(1) / x)
    x = np.concatenate([(rng.random(n) - 0.5) * 200, (rng.random(n // 8) - 0.5) * 20]).astype(np.float32)
    unit("B4_ggml_v_expf", 3, x, E.v_expf(x))
    x = np.exp(rng.uniform(np.log(1e-12), np.log(1e8), n)).astype(np.float32)
    unit("B5_sqrt_cr", 4, x, np.sqrt(x))
    x = np.concatenate([(rng.random(n) - 0.5) * 40, rng.standard_normal(n // 4) * 2]).astype(np.float32)
    unit("B6_silu", 5, x, E.swiglu(x, np.ones_like(x)))
    X = (rng.standard_normal((rows, 1024)) * np.exp(rng.uniform(-6, 3, (rows, 1)))).astype(np.float32)
    X[::7, rng.integers(0, 1024, rows // 7 + 1)[: len(X[::7])]] *= 3000
    Wn = (1 + rng.standard_normal(1024) * 0.3).astype(np.float32)
    prm = np.array([1024, rows, EPS.view(np.int32), 1], np.int32)
    P.run("B", "B7_rms_norm", "ggml_rms_norm_mul.comp", rows, [(0, P.put(prm)), (1, P.put(X)), (2, P.put(Wn))], 3,
          rows * 1024 * 4, stage="B7_rms_norm", ref=(E.rms_norm(X, 1e-6) * Wn).astype(np.float32))


def part_c(P, w, dump, nsteps):
    qt = lambda name: 4 if w[name + ".type"] == "Q4_K" else 6  # noqa: E731
    kpacks, vpacks = [], []
    for s in range(nsteps):
        d = os.path.join(dump, f"step{s}")
        rows = manifest(d)
        n = [0]

        def run(tag, shader, groups, ins, ob, nbytes, ints=(), stage=None):
            n[0] += 1
            ref = None
            if stage is not None:
                oname, oop = next((o, op) for ours, o, op in STAGES if ours == stage)
                ref = oracle(d, rows, oname, oop)
            return P.run("C", f"C{s}_{n[0]:02d}_{tag}", shader, groups, ins, ob, nbytes, ints, step=s,
                         stage=stage or tag, ref=ref)

        def rms(x, nrows, nn, wt, stage):
            prm = P.put(np.array([nn, nrows, EPS.view(np.int32), 1], np.int32))
            return run(stage, "ggml_rms_norm_mul.comp", nrows, [(0, prm), (1, x), (2, P.put(wt))], 3, nrows * nn * 4,
                       stage=stage)

        def matvec(name, x, stage):
            raw = w[name]
            nrow, nb = raw.shape[:2]
            q8 = run(stage + "_q8k", "ggml_q8k_quantize.comp", nb, [(0, x)], 1, nb * 273 * 4)
            prm = P.put(np.array([nrow, nb, qt(name)], np.int32))
            return run(stage, "ggml_matvec_qk.comp", (nrow + 63) // 64, [(0, prm), (1, P.put(raw)), (2, q8)], 3,
                       nrow * 4, stage=stage)

        def rope(x, heads, tab, stage):
            prm = P.put(np.array([heads, HD], np.int32))
            return run(stage, "ggml_rope_neox.comp", heads, [(0, prm), (1, x), (2, tab)], 3, heads * HD * 4, stage=stage)

        def elt(mode, a, b, nn, stage=None, tag=None):
            nout = nn // 2 if mode == 2 else nn
            return run(tag or stage, "ggml_eltwise.comp", (nn + 63) // 64, [(0, a), (1, b)], 2, nout * 4,
                       [("mode", mode), ("n", nn)], stage=stage)

        x = P.put(oracle(d, rows, "embd", "GET_ROWS"))
        a = rms(x, 1, N_EMBD, w["attn_norm"], "attn_norm")
        q = matvec("attn_q", a, "Qcur")
        k = matvec("attn_k", a, "Kcur")
        v = matvec("attn_v", a, "Vcur")
        qn = rms(q, N_HEAD, HD, w["attn_q_norm"], "Qcur_normed")
        kn = rms(k, N_HEAD_KV, HD, w["attn_k_norm"], "Kcur_normed")
        cos, sin = E.rope_neox_cache(s, HD, 1e6)
        tab = P.put(np.concatenate([cos, sin]).astype(np.float32))
        qr = rope(qn, N_HEAD, tab, "Qrope")
        kr = rope(kn, N_HEAD_KV, tab, "Krope")
        kpacks.append(elt(2, kr, kr, N_HEAD_KV * HD, tag="kpack"))
        vpacks.append(elt(2, v, v, N_HEAD_KV * HD, tag="vpack"))
        kc, vc = f"$O/C{s}_kcache.bin", f"$O/C{s}_vcache.bin"
        P.lines["C"].append(f"concat {kc} " + " ".join(kpacks))
        P.lines["C"].append(f"concat {vc} " + " ".join(vpacks))
        scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(HD)))
        prm = P.put(np.array([s + 1, N_HEAD, N_HEAD_KV, scale.view(np.int32)], np.int32))
        o = run("kqv_out", "ggml_fa_f16.comp", N_HEAD, [(0, prm), (1, qr), (2, kc), (3, vc)], 4, N_HEAD * HD * 4,
                stage="kqv_out")
        ao = matvec("attn_output", o, "attn_out")
        fi = elt(0, ao, x, N_EMBD, stage="ffn_inp")
        fn = rms(fi, 1, N_EMBD, w["ffn_norm"], "ffn_norm")
        gt = matvec("ffn_gate", fn, "ffn_gate")
        up = matvec("ffn_up", fn, "ffn_up")
        sw = elt(1, gt, up, 3072, stage="ffn_swiglu")
        fo = matvec("ffn_down", sw, "ffn_out")
        elt(0, fo, fi, N_EMBD, stage="l_out")
        P.lines["C"].append(f"checkpoint C_step{s}")


def main():
    model, dump, nsteps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    P = Plan(out)
    part_b(P)
    part_c(P, E.load_weights(model, os.environ.get("LLAMA_CPP")), dump, nsteps)
    P.write()
    nb = sum(1 for r in P.labels if r[1] == "B")
    print(f"plan: B {nb} dispatches, C {len(P.labels) - nb} dispatches over {nsteps} steps; "
          f"{len(os.listdir(os.path.join(out, 'in')))} input files")


if __name__ == "__main__":
    main()
