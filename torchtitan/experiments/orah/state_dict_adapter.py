# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.models.base.state_dict_adapter import StateDictAdapter
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

from .siglep import SIGLEP_HF_KEY_REPLACEMENTS


_PROJECTOR_KEYS = {
    "projector.0.weight": "vision_encoder.projector.linear_fc1.weight",
    "projector.0.bias": "vision_encoder.projector.linear_fc1.bias",
    "projector.2.weight": "vision_encoder.projector.linear_fc2.weight",
    "projector.2.bias": "vision_encoder.projector.linear_fc2.bias",
}


class OrahStateDictAdapter(StateDictAdapter):
    """Read model weights from a full ELM Orah checkpoint."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self.decoder = Qwen35StateDictAdapter(model_config, None)

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        decoder = self.decoder.to_hf(
            {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("vision_encoder.")
            },
            to_quantized_layout=to_quantized_layout,
        )
        result = {
            "language_model." + key.replace("model.language_model.", "model.", 1): value
            for key, value in decoder.items()
        }
        for key, value in state_dict.items():
            if key.startswith("vision_encoder.") and not key.startswith(
                "vision_encoder.projector."
            ):
                name = key.removeprefix("vision_encoder.")
                for hf_name, native_name in reversed(
                    SIGLEP_HF_KEY_REPLACEMENTS.items()
                ):
                    name = name.replace(native_name, hf_name)
                result[f"encoder.model.{name}"] = value
        result.update(
            {hf_key: state_dict[key] for hf_key, key in _PROJECTOR_KEYS.items()}
        )
        return result

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        decoder = {
            key.removeprefix("language_model.").replace(
                "model.", "model.language_model.", 1
            ): value
            for key, value in hf_state_dict.items()
            if key.startswith("language_model.")
        }
        result = self.decoder.from_hf(decoder, is_quantized_layout=is_quantized_layout)
        for key, value in hf_state_dict.items():
            if key.startswith("encoder.model."):
                name = key.removeprefix("encoder.model.")
                for hf_name, native_name in SIGLEP_HF_KEY_REPLACEMENTS.items():
                    name = name.replace(hf_name, native_name)
                result[f"vision_encoder.{name}"] = value
        result.update(
            {key: hf_state_dict[hf_key] for hf_key, key in _PROJECTOR_KEYS.items()}
        )
        return result
