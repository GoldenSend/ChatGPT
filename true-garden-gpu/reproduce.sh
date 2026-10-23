#!/usr/bin/env bash
# Rebuild every offline result in RESULTS/ from scratch on an x86-64 Linux host
# with Mesa (LLVMPipe), EGL/GLES dev headers, gcc/g++, cmake, python3 + numpy.
# Never touches a PS4. Takes ~10 minutes on 4 cores (mostly the llama.cpp build).
#
#   WORK=/some/scratch/dir ./reproduce.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/.work}"
LLAMA_COMMIT=23b0202a189c44a54625aadcb37a946dd1d6278d
mkdir -p "$WORK"/{models,dumps,bin}
export LLAMA_CPP="$WORK/llama.cpp" WORKDIR="$WORK"

# 1. llama.cpp at the PS4 CPU baseline, built for AMD Jaguar (AVX+F16C, no AVX2/FMA)
if [ ! -d "$LLAMA_CPP" ]; then
    git clone -q https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
fi
git -C "$LLAMA_CPP" checkout -q "$LLAMA_COMMIT"
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build-jaguar" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_F16C=ON -DGGML_SSE42=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF \
    -DGGML_BMI2=OFF -DGGML_AVX512=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DCMAKE_C_FLAGS="-march=btver2" -DCMAKE_CXX_FLAGS="-march=btver2" > "$WORK/cmake.log"
cmake --build "$LLAMA_CPP/build-jaguar" -j"$(nproc)" --target llama llama-quantize > "$WORK/build.log"
B="$LLAMA_CPP/build-jaguar/bin"

# 2. tools
g++ -O2 -march=btver2 -std=c++17 -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" "$HERE/tools/dump_layer0.cpp" \
    -L"$B" -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$B" -o "$WORK/bin/dump_layer0"
gcc -O2 -o "$WORK/bin/gles_run" "$HERE/tools/gles_run.c" -lEGL -lGLESv2

# 3. synthetic Qwen3-0.6B-shaped 1-layer models (deterministic; hashes in BASELINE.md)
for s in 1 2 3; do
    python3 "$HERE/tools/make_synth_qwen3.py" "$WORK/models/synth-f32-s$s.gguf" --seed $s
    "$B/llama-quantize" "$WORK/models/synth-f32-s$s.gguf" "$WORK/models/synth-q4km-s$s.gguf" Q4_K_M > /dev/null 2>&1
done
sha256sum "$WORK"/models/*.gguf

# 4. oracle dumps (FA=1 is llama.cpp's CPU default; thread count does not change results)
D="$WORK/bin/dump_layer0"
"$D" "$WORK/models/synth-q4km-s1.gguf" "$WORK/dumps/s1_fa1_t1" 1 1 17 301 > /dev/null 2>&1
"$D" "$WORK/models/synth-q4km-s2.gguf" "$WORK/dumps/s2_fa1" 1 4 5 77 300 411 9 > /dev/null 2>&1
"$D" "$WORK/models/synth-q4km-s3.gguf" "$WORK/dumps/s3_fa1" 1 4 499 0 1 2 > /dev/null 2>&1
LONG=$(python3 -c "import numpy as np; print(' '.join(map(str, np.random.default_rng(9).integers(0, 512, 48))))")
"$D" "$WORK/models/synth-q4km-s1.gguf" "$WORK/dumps/s1_long" 1 4 $LONG > /dev/null 2>&1

# 5. tests (a negative control must FAIL; if one passes, the test has lost its teeth and the script stops)
expect_fail() {
    if "$@" > "$WORK/negctl.log"; then echo "NEGATIVE CONTROL PASSED (it must fail): $*"; exit 1; fi
    echo "negative control, failed as required: $(tail -1 "$WORK/negctl.log")"
}
G="$WORK/bin/gles_run"
M="$WORK/models"
DD="$WORK/dumps"
python3 "$HERE/TESTS/test_emu_vs_oracle.py" "$M/synth-q4km-s1.gguf" "$DD/s1_fa1_t1" 17 301 | tail -1
python3 "$HERE/TESTS/test_emu_vs_oracle.py" "$M/synth-q4km-s2.gguf" "$DD/s2_fa1" 5 77 300 411 9 | tail -1
python3 "$HERE/TESTS/test_emu_vs_oracle.py" "$M/synth-q4km-s3.gguf" "$DD/s3_fa1" 499 0 1 2 | tail -1
python3 "$HERE/TESTS/test_gles_exact_lib.py" "$G" "$WORK" | tail -1
python3 "$HERE/TESTS/test_gles_fa_vs_oracle.py" "$G" "$DD/s1_fa1_t1" "$DD/s2_fa1" "$DD/s3_fa1" "$DD/s1_long" | tail -1
expect_fail python3 "$HERE/TESTS/test_gles_fa_vs_oracle.py" --negative-control "$G" "$DD/s1_fa1_t1" "$DD/s2_fa1" "$DD/s3_fa1" "$DD/s1_long"
for m in "s1 s1_fa1_t1" "s2 s2_fa1" "s3 s3_fa1" "s1 s1_long"; do
    set -- $m
    python3 "$HERE/TESTS/test_gles_matvec_vs_oracle.py" "$G" "$M/synth-q4km-$1.gguf" "$DD/$2" | tail -1
    python3 "$HERE/TESTS/test_gles_layer_vs_oracle.py" "$G" "$M/synth-q4km-$1.gguf" "$DD/$2" | tail -1
done
expect_fail python3 "$HERE/TESTS/test_gles_matvec_vs_oracle.py" --negative-control "$G" "$M/synth-q4km-s1.gguf" "$DD/s1_fa1_t1"
python3 "$HERE/experiments/sensitivity.py" "$WORK/sensitivity.json" "$M/synth-q4km-s1.gguf" "$M/synth-q4km-s2.gguf" "$M/synth-q4km-s3.gguf" > /dev/null
python3 "$HERE/experiments/summarize_sensitivity.py" "$WORK/sensitivity.json" "$WORK/sensitivity_summary.md"
echo "sensitivity summary: $WORK/sensitivity_summary.md"
echo "ALL OFFLINE CHECKS DONE"
