# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Orah's ECG embedding replacement on the existing Qwen3.5 decoder."""

# Shape suffixes: B = ECGs, C = leads, L = samples, T = packed text tokens,
# N = packed ECG tokens, D = LLM width.

from dataclasses import dataclass, fields, replace

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
)
from torchtitan.models.qwen3_5 import build_model_config as build_qwen35_config
from torchtitan.models.qwen3_5.model import Qwen35Model

from .siglep import projected_encoder_config, ProjectedSigLEP, SigLEP


class OrahModel(Qwen35Model):
    """Qwen3.5 with a SigLEP tower in place of the image/video encoder.

    The trainer supplies token-major inputs and ECGs shaped (batch, 12, 2500).
    Each ECG replaces num_ecg_tokens placeholders, in ECG batch order.
    """

    # Qwen's checkpoint adapter describes its image tower, not SigLEP.
    state_dict_adapter_cls = None

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        vision_encoder: ProjectedSigLEP.Config
        ecg_token_id: int
        trainable: tuple[str, ...] = ("projector", "language_model")

        def __post_init__(self):
            if not 0 <= self.ecg_token_id < self.vocab_size:
                raise ValueError("ecg_token_id must be within the language vocabulary")
            unknown = set(self.trainable) - {"encoder", "projector", "language_model"}
            if unknown:
                raise ValueError(f"Unknown trainable components: {sorted(unknown)}")

    def __init__(self, config: Config):
        # Reuse Qwen's construction and TP validation without building its ViT.
        super().__init__(replace(config, vision_encoder=None))
        if "language_model" not in config.trainable:
            self.requires_grad_(False)
        self.config = config
        self.vision_encoder = config.vision_encoder.build()
        self.vision_encoder.requires_grad_("encoder" in config.trainable)
        self.vision_encoder.projector.requires_grad_("projector" in config.trainable)

    def preprocess_inputs(self, input_dict, **kwargs):
        input_dict = dict(input_dict)
        ecg_values = input_dict.pop("ecg_values", None)
        if self.tok_embeddings is not None and ecg_values is not None:
            self._validate_placeholders(
                input_dict["input"], ecg_values, input_dict.get("positions")
            )
            # Adapt to Qwen's existing CP sharding and GDN metadata path.
            input_dict["pixel_values"] = ecg_values
            input_dict["special_tokens"] = {"image_id": self.config.ecg_token_id}
        return super().preprocess_inputs(input_dict, **kwargs)

    def _validate_placeholders(self, tokens_T, ecg_values, positions_T=None):
        if tokens_T.ndim != 1:
            raise ValueError(
                "Orah expects TorchTitan's flattened, token-major text input"
            )
        expected = ecg_values.shape[0] * self.config.vision_encoder.num_ecg_tokens
        actual = int(tokens_T.eq(self.config.ecg_token_id).sum())
        if actual != expected:
            raise ValueError(
                f"Expected {expected} ECG placeholders for {ecg_values.shape[0]} ECGs, "
                f"got {actual}"
            )
        if positions_T is not None:
            document_ids_T = (positions_T == 0).cumsum(0) - 1
            counts_B = torch.bincount(
                document_ids_T[tokens_T == self.config.ecg_token_id]
            )
            if torch.any(
                (counts_B != 0)
                & (counts_B != self.config.vision_encoder.num_ecg_tokens)
            ):
                raise ValueError(
                    "Each ECG-bearing document must contain num_ecg_tokens placeholders"
                )

    def _prepare_multimodal_embeds(
        self,
        tokens,
        *,
        pixel_values,
        image_vision_bank_indices_T,
        **kwargs,
    ):
        del kwargs
        inputs_TD = self.tok_embeddings(tokens)
        # The inherited embedding hook calls its modality input pixel_values.
        ecg_values = pixel_values
        if ecg_values is None:
            raise ValueError("Orah requires ecg_values with shape (batch, 12, 2500)")
        if image_vision_bank_indices_T is None:
            self._validate_placeholders(tokens, ecg_values)
            image_vision_bank_indices_T = build_vision_bank_indices(
                tokens, placeholder_id=self.config.ecg_token_id
            )
        features_ND = self.vision_encoder(ecg_values)
        return gather_vision_embeds(
            inputs_TD,
            vision_bank_VD=features_ND,
            vision_bank_indices_T=image_vision_bank_indices_T,
        )

    def forward(self, tokens, *, ecg_values=None, **kwargs):
        if ecg_values is not None:
            kwargs["pixel_values"] = ecg_values
        return super().forward(tokens, **kwargs)


def build_model_config(
    flavor: str = "9B",
    *,
    encoder: SigLEP.Config,
    ecg_token_id: int,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    trainable: tuple[str, ...] = ("projector", "language_model"),
    lora: LoRATransform | None = None,
    vocab_size: int | None = None,
) -> OrahModel.Config:
    """Reuse Qwen's recipe and apply native LoRA only to its language modules."""
    decoder = build_qwen35_config(
        flavor, parallelism=parallelism, seq_len=seq_len, attn_backend=attn_backend
    )
    decoder.vision_encoder = None
    if vocab_size is not None:
        decoder.vocab_size = vocab_size
        decoder.tok_embeddings = replace(
            decoder.tok_embeddings, num_embeddings=vocab_size
        )
        decoder.lm_head = replace(decoder.lm_head, out_features=vocab_size)
    if lora is not None:
        decoder = lora.transform(decoder)
    return OrahModel.Config(
        **{
            f.name: getattr(decoder, f.name)
            for f in fields(Qwen35Model.Config)
            if f.name != "vision_encoder"
        },
        vision_encoder=projected_encoder_config(encoder, output_size=decoder.dim),
        ecg_token_id=ecg_token_id,
        trainable=trainable,
    )
