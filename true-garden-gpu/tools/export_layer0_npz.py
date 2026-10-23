#!/usr/bin/env python3
"""Write a model's layer-0 tensors (raw quant blocks, as emu/ggml_emu.py:
load_weights reads them) to an .npz, so a machine with only numpy (the PS4)
can run the GPU tests without gguf-py and its PyYAML dependency. The token
embedding and output matrix are left out (the layer test takes the embedding
row from the llama.cpp dump).

usage: LLAMA_CPP=/path/to/llama.cpp export_layer0_npz.py MODEL.gguf OUT.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emu"))
import ggml_emu as E  # noqa: E402

w = E.load_weights(sys.argv[1], os.environ.get("LLAMA_CPP"))
keep = {k: v for k, v in w.items() if not k.startswith(("blk.", "token_embd", "output"))}
np.savez(sys.argv[2], **{k: (np.array(v) if isinstance(v, str) else v) for k, v in keep.items()})
print(f"{sys.argv[2]}: {len(keep)} arrays")
