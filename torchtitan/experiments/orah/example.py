# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Orah's debug decoder on one CUDA GPU, optionally with a pretrained ECG tower.

From the repository root: python -m torchtitan.experiments.orah.example
Add --siglip2-checkpoint /path/to/best.pt for an ECG-adapted
google/siglip2-base-patch16-naflex checkpoint trained with patch_size=25.
"""

# Shape suffixes: B = batch, C = ECG leads, L = ECG samples,
# S = text sequence length, T = flattened text tokens, V = vocabulary.

import argparse
from dataclasses import replace

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.orah import build_model_config, SigLEP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--siglip2-checkpoint",
        help="ECG SigLIP2 base NaFlex .pt checkpoint containing model_state_dict",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "This example requires CUDA for Qwen3.5's Gated DeltaNet kernels."
        )

    torch.manual_seed(42)
    device = torch.device("cuda")
    batch_size, seq_len, vocab_size = 2, 128, 1024
    ecg_token_id = 1
    encoder = SigLEP.Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_leads=12,
        segment_length=2500,
        patch_size=25,
        num_ecg_tokens=100,
    )
    if args.siglip2_checkpoint is not None:
        encoder = replace(
            encoder,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
    config = build_model_config(
        "debugmodel",
        encoder=encoder,
        ecg_token_id=ecg_token_id,
        parallelism=ParallelismConfig(),
        seq_len=seq_len,
        vocab_size=vocab_size,
        attn_backend="flex",
    )
    model = config.build().to(device=device, dtype=torch.bfloat16)
    model.init_states(buffer_device=device)
    if args.siglip2_checkpoint is not None:
        model.vision_encoder.load_siglip2_checkpoint(args.siglip2_checkpoint)
        print(f"Loaded ECG encoder from {args.siglip2_checkpoint}")
    model.eval()

    ecg_BCL = torch.randn(batch_size, 12, 2500, device=device)
    tokens_BS = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    tokens_BS[:, : encoder.num_ecg_tokens] = ecg_token_id
    tokens_T = tokens_BS.flatten()
    # Reset positions at each sample so attention cannot cross sample boundaries.
    positions_T = torch.arange(seq_len, device=device).repeat(batch_size)
    attention_metadata = model._get_attention_metadata(positions_T)

    with torch.no_grad():
        logits_TV = model(
            tokens_T,
            ecg_values=ecg_BCL,
            positions=positions_T,
            attention_metadata=attention_metadata,
        )

    print(f"ECGs: {tuple(ecg_BCL.shape)}")
    print(f"Token IDs: {tuple(tokens_BS.shape)} -> {tuple(tokens_T.shape)}")
    print(f"Logits: {tuple(logits_TV.shape)}")


if __name__ == "__main__":
    main()
