# Tests

All tests compare against **llama.cpp's own tensors** (dumped by `tools/dump_layer0`), bit for bit, except `test_gles_exact_lib.py`, which compares the GLSL helpers with CPU reference implementations. `../reproduce.sh` runs them all; the committed outputs are in `../RESULTS/`.

Environment: `LLAMA_CPP` points at the llama.cpp checkout (for `gguf-py`); `WORKDIR` is a scratch directory for buffers (default `/tmp`).

| Test | Checks | Pass condition | Negative control |
|---|---|---|---|
| `test_emu_vs_oracle.py MODEL DUMPDIR TOK...` | the numpy emulator against every layer-0 stage of llama.cpp | every stage bit-identical | (the sensitivity study plays this role: each GPU-style variant must break it) |
| `test_gles_exact_lib.py GLES_RUN WORKDIR [N]` | FP16 RNE, glibc `expf`, correctly rounded rcp/sqrt, `ggml_v_expf`, SiLU, RMSNorm stress rows (with ~3000x outliers) on the current GLES driver | 0 mismatches, except `expf` cases where glibc itself is not correctly rounded (listed, ~4e-7 per call) | none built in; any driver contraction or denormal flush shows up as mismatches |
| `test_gles_fa_vs_oracle.py [--negative-control] GLES_RUN DUMPDIR...` | flash attention shader with llama.cpp's own Q and FP16 K/V cache against `FLASH_ATTN_EXT` | every step bit-identical | `--negative-control` swaps FP16 RNE for round-toward-zero (the hardware's apparent `packHalf2x16` behaviour): n_kv = 1 must still pass and every n_kv >= 2 step must fail |
| `test_gles_matvec_vs_oracle.py [--negative-control] GLES_RUN MODEL DUMPDIR` | Q8_K quantization + Q4_K/Q6_K matvec for all 7 projections against `MUL_MAT`; GPU Q8_K blocks against the emulator | every matvec bit-identical, Q8_K blocks identical | `--negative-control` folds ggml's 8 lanes into one row-serial accumulator: matvecs must differ |
| `test_gles_layer_vs_oracle.py GLES_RUN MODEL DUMPDIR` | the whole layer chained on the GPU driver: only the embedding row and the RoPE table come from the host; the K/V cache is built from the shaders' own outputs | all 17 stages bit-identical at every step | inherits the two controls above |

Dumps are produced with:

```sh
dump_layer0 MODEL.gguf OUTDIR 1 4 TOKEN0 TOKEN1 ...   # FA=1 (llama.cpp's CPU default), 4 threads
```

`OUTDIR/stepN/` holds one `.bin` per tensor and a `MANIFEST.tsv` (index, name, op, type, ne0..3, sources).

## Using real Qwen3-0.6B weights

The tests read layer-0 tensor types from the GGUF, so they work on the real `Qwen3-0.6B-Q4_K_M.gguf` (sha `b0638f08...`) as long as layer 0 is Q4_K/Q6_K and the token embedding is Q6_K (true for the synthetic tied-embedding files made by the same `llama-quantize`; expected but not checked for the real file; `emu/ggml_emu.py: load_weights` raises a clear error on any other layer-0 type). A full-model dump holds all 28 layers; the tests pick layer 0 by name and graph order. This was **not run here** (no access to real weights); it is the first job in `../HANDOFF-CODEX.md`.

## What a PASS means and does not mean

A PASS on LLVMPipe means the shaders implement llama.cpp's arithmetic exactly on an IEEE-conformant driver. It says nothing about Liverpool until `test_gles_exact_lib.py` and the layer test pass on radeonsi on the PS4 under the hardware gate.
