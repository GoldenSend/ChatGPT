#!/usr/bin/env bash
# Build the GPU-022 bundle on a host with numpy, gguf-py and Mesa LLVMPipe:
#   1. tools/make_gate_plan.py writes the plan, host inputs and references
#   2. the plan is replayed through tools/gpu022_gate.sh on LLVMPipe
#   3. every replay output that has a reference must equal it (llama.cpp tensors
#      for layer 0, CPU references for the helpers); the verified outputs become
#      expected/
#   4. SHA256SUMS, then a full rehearsal of the gate (A, B, C) from the finished
#      bundle, then a .tar.gz
# usage: LLAMA_CPP=/path/llama.cpp build_gate_bundle.sh MODEL.gguf DUMPDIR NSTEPS OUT.tar.gz
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=$1; DUMP=$2; NSTEPS=$3; OUT=$(realpath -m "$4")
T=$(mktemp -d); BND="$T/gpu022"
python3 "$HERE/tools/make_gate_plan.py" "$MODEL" "$DUMP" "$NSTEPS" "$T/gen"
mkdir -p "$BND/tools" "$BND/shaders"
cp "$HERE"/shaders/* "$BND/shaders/"
cp "$HERE"/tools/{gles_run.py,gles_run_guarded.sh,compile_check.sh,gpu022_gate.sh,gate_compare.py} "$BND/tools/"
cp -r "$T/gen/plan" "$T/gen/in" "$BND/"
BID=$(cat /proc/sys/kernel/random/boot_id)
export LIBGL_ALWAYS_SOFTWARE=1
GATE_RECORD=1 ALLOW_SOFTWARE=1 "$BND/tools/gpu022_gate.sh" "$BID" BC > "$T/record.log"
R=$(ls -d "$BND"/gate-out-*/out)
python3 - "$R" "$T/gen/refs" "$BND/plan/labels.tsv" <<'PY'
import os, sys
import numpy as np
out, refs, labels = sys.argv[1:4]
n = bad = 0
for line in open(labels):
    label, part, step, stage, kind = line.rstrip("\n").split("\t")
    if kind != "ref":
        continue
    g = np.fromfile(os.path.join(out, label + ".bin"), np.float32)
    r = np.fromfile(os.path.join(refs, label + ".bin"), np.float32)
    d = (g.view(np.uint32) != r.view(np.uint32)) & ~(np.isnan(g) & np.isnan(r)) if g.size == r.size else np.ones(1, bool)
    alt = os.path.join(refs, label + ".alt.bin")
    if os.path.exists(alt) and g.size == r.size:  # documented known class only
        d &= g.view(np.uint32) != np.fromfile(alt, np.float32).view(np.uint32)
    n += 1
    if d.any():
        bad += 1
        print(f"REFERENCE MISMATCH {label}: {int(d.sum())} values")
print(f"replay vs references (llama.cpp tensors for layer 0, CPU references for helpers): {n - bad}/{n} bit-exact")
sys.exit(1 if bad else 0)
PY
mkdir -p "$BND/expected"
cut -f1 "$BND/plan/labels.tsv" | while read -r l; do cp "$R/$l.bin" "$BND/expected/"; done
rm -rf "$BND"/gate-out-*
(cd "$BND" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
ALLOW_SOFTWARE=1 "$BND/tools/gpu022_gate.sh" "$BID" ABC > "$T/rehearsal.log" 2>&1 || { tail -20 "$T/rehearsal.log"; exit 1; }
grep -E "COMPARE|GATE RESULT|GL_RENDERER" "$T/rehearsal.log"
rm -rf "$BND"/gate-out-*
tar -C "$T" -czf "$OUT" gpu022
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)), $(wc -l < "$BND/SHA256SUMS") files, sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"
rm -rf "$T"
