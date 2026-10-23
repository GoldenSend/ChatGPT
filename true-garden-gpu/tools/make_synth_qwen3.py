#!/usr/bin/env python3
"""Write a 1-layer, Qwen3-0.6B-shaped GGUF with synthetic F32 weights.

The real Qwen3-0.6B weights are not reachable from the cloud container
(Hugging Face is blocked by the network policy), so this model has the exact
layer-0 *geometry* of Qwen3-0.6B (n_embd 1024, 16 q heads, 8 kv heads,
head_dim 128, n_ff 3072, rope base 1e6, rms eps 1e-6, tied embeddings) but
random weights. Quantize it with llama-quantize Q4_K_M afterwards so layer 0
gets the same tensor types as the real file (Q4_K q/k/o/gate/up, Q6_K v/down).

Usage: make_synth_qwen3.py OUT.gguf [--seed N] [--vocab N]
"""
import argparse
import os
import sys

import numpy as np

LLAMA_CPP = os.environ.get("LLAMA_CPP")
if LLAMA_CPP:
    sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))
import gguf  # noqa: E402

N_EMBD, N_HEAD, N_HEAD_KV, HEAD_DIM, N_FF = 1024, 16, 8, 128, 3072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=512)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    def g(shape, std):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    w = gguf.GGUFWriter(a.out, "qwen3")
    w.add_context_length(4096)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(1)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_freq_base(1_000_000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_vocab_size(a.vocab)
    w.add_tokenizer_model("none")
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # numpy shape is (rows, cols) = reversed ggml ne
    w.add_tensor("token_embd.weight", g((a.vocab, N_EMBD), 0.03))
    w.add_tensor("output_norm.weight", (1.0 + g((N_EMBD,), 0.2)))
    w.add_tensor("blk.0.attn_norm.weight", np.abs(1.0 + g((N_EMBD,), 0.3)))
    w.add_tensor("blk.0.attn_q.weight", g((N_HEAD * HEAD_DIM, N_EMBD), 0.03))
    w.add_tensor("blk.0.attn_k.weight", g((N_HEAD_KV * HEAD_DIM, N_EMBD), 0.03))
    w.add_tensor("blk.0.attn_v.weight", g((N_HEAD_KV * HEAD_DIM, N_EMBD), 0.03))
    w.add_tensor("blk.0.attn_output.weight", g((N_EMBD, N_HEAD * HEAD_DIM), 0.03))
    w.add_tensor("blk.0.attn_q_norm.weight", (1.0 + g((HEAD_DIM,), 0.2)))
    w.add_tensor("blk.0.attn_k_norm.weight", (1.0 + g((HEAD_DIM,), 0.2)))
    w.add_tensor("blk.0.ffn_norm.weight", np.abs(1.0 + g((N_EMBD,), 0.3)))
    w.add_tensor("blk.0.ffn_gate.weight", g((N_FF, N_EMBD), 0.03))
    w.add_tensor("blk.0.ffn_up.weight", g((N_FF, N_EMBD), 0.03))
    w.add_tensor("blk.0.ffn_down.weight", g((N_EMBD, N_FF), 0.02))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
