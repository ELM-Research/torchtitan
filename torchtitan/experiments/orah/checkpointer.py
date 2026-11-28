# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from torchtitan.components.checkpointer import CheckpointManager, MODEL
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter


logger = logging.getLogger(__name__)
_SIGLIP2_BASE = "google/siglip2-base-patch16-naflex"


class _QwenBaseReader(dcp.HuggingFaceStorageReader):
    def __init__(self, path: str, *, vocab_size: int):
        super().__init__(path, thread_count=2)
        self.vocab_size = vocab_size

    def read_metadata(self) -> Metadata:
        metadata = super().read_metadata()
        for key in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            tensor = metadata.state_dict_metadata.get(key)
            if (
                isinstance(tensor, TensorStorageMetadata)
                and tensor.size[0] > self.vocab_size
            ):
                # Orah trims the padded Qwen vocabulary. Keep the on-disk chunk
                # metadata so DCP can read each rank's slice directly.
                tensor.size = torch.Size((self.vocab_size, *tensor.size[1:]))
        return metadata


class OrahCheckpointer(CheckpointManager):
    """Use public pretrained weights when there is no Orah checkpoint to load."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        qwen_model: str
        siglep_checkpoint: str | None = None
        additional_save_steps: tuple[int, ...] = ()

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.qwen_model = config.qwen_model
        self.siglep_checkpoint = config.siglep_checkpoint
        self.additional_save_steps = frozenset(config.additional_save_steps)

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        return super()._should_save(curr_step, last_step) or (
            not self.load_only and curr_step in self.additional_save_steps
        )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        if super().load(step):
            return True

        model = self.states[MODEL]
        (part,) = model.model
        decoder = Qwen35StateDictAdapter(part.config, hf_assets_path=None)
        language = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("vision_encoder.")
        }
        qwen_weights = decoder.to_hf(language)
        qwen_path = snapshot_download(self.qwen_model, allow_patterns="*.safetensors")
        logger.info("Loading Qwen language weights from %s", self.qwen_model)
        dcp.load(
            qwen_weights,
            storage_reader=_QwenBaseReader(
                qwen_path, vocab_size=language["tok_embeddings.weight"].size(0)
            ),
        )
        pretrained = decoder.from_hf(qwen_weights)
        missing = language.keys() - pretrained.keys()
        if missing:
            raise ValueError(
                f"Qwen checkpoint is missing Orah language weights: {sorted(missing)}"
            )
        model.load_state_dict(pretrained)
        logger.info("Loaded Qwen language weights from %s", self.qwen_model)

        if self.siglep_checkpoint is None:
            vision_path = snapshot_download(
                _SIGLIP2_BASE, allow_patterns="model.safetensors"
            )
            part.vision_encoder.load_siglip2_hf(vision_path, ecg_embeddings=False)
            logger.info("Loaded SigLIP2 vision weights from %s", _SIGLIP2_BASE)
        elif Path(self.siglep_checkpoint).is_dir():
            part.vision_encoder.load_siglip2_hf(
                self.siglep_checkpoint, ecg_embeddings=True
            )
            logger.info("Loaded SigLEP weights from %s", self.siglep_checkpoint)
        else:
            part.vision_encoder.load_siglip2_checkpoint(self.siglep_checkpoint)
            logger.info("Loaded SigLEP weights from %s", self.siglep_checkpoint)
        return True
