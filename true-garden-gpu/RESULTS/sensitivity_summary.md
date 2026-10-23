| Variant (one op computed GPU-style) | token0 fails | token1 fails | token1 first differing stage | token1 Q8_K flips at gate/up input (median) | token1 max abs pre-FFN diff | token1 bad l_out @1e-5 (median of fails) |
|---|---|---|---|---|---|---|
| exact (control) | 0/12 | 0/12 | - | 0 | 0.00e+00 | 0 |
| rms_norm fp32 tree | 0/12 | 1/12 | Qcur_normed | 0 | 3.10e-05 | 1023 |
| matvec row-serial | 0/12 | 6/12 | Qcur | 0 | 1.82e-03 | 1023 |
| rope direct theta | 0/12 | 0/12 | Qrope | 0 | 3.34e-06 | 0 |
| rope hw-trig model | 0/12 | 1/12 | Qrope | 0 | 4.29e-06 | 997 |
| FA ggml + CR exp | 0/12 | 0/12 | - | 0 | 0.00e+00 | 0 |
| attn fp32 standard | 0/12 | 12/12 | kqv_out | 72 | 1.11e-02 | 1023 |
| FA: Q kept fp32 | 0/12 | 12/12 | kqv_out | 41.5 | 1.02e-02 | 1023 |
| FA: KQ dot serial | 0/12 | 1/12 | kqv_out | 0 | 1.43e-06 | 997 |
| FA: fp32 accumulator | 0/12 | 12/12 | kqv_out | 62 | 9.44e-03 | 1023.5 |
| silu precise exp | 0/12 | 0/12 | ffn_swiglu | 0 | 0.00e+00 | 0 |
| all precise | 0/12 | 12/12 | Qcur | 72 | 1.11e-02 | 1023.5 |

Single Q8_K code flip probe (token 1, exact pipeline, one `ffn_norm` element moved across its rounding boundary):

| model | input change | codes changed | downstream Q8_K flips at down-proj input | l_out values changed | l_out max abs | bad @1e-5 |
|---|---|---|---|---|---|---|
| synth-q4km-s1.gguf | 5.13e-06 | 1 | 49 | 1024/1024 | 1.43e-02 | 1024 |
| synth-q4km-s2.gguf | 3.28e-06 | 1 | 69 | 1024/1024 | 1.30e-02 | 1022 |
| synth-q4km-s3.gguf | 1.70e-06 | 1 | 47 | 1024/1024 | 1.59e-02 | 1024 |
