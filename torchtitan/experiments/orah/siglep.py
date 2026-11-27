# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SigLEP's ECG tower, using TorchTitan's shared vision transformer blocks."""

# Shape suffixes: B = ECGs, C = leads, L = samples, P = patches per ECG,
# K = leads times patch width, T = packed patches, D = hidden size,
# N = output tokens per ECG, H = heads, Q = head width.

from dataclasses import dataclass, fields
from functools import partial

import spmd_types as spmd
import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from spmd_types import SpmdType
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.distributed.spmd_types import disable_dense_sp, spmd_mesh_size
from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import ColumnParallelLinear, Linear, RowParallelLinear
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.flops import active_parameter_flops_per_token
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
    vision_attention_sharding_config,
    vision_colwise_sharding_config,
    vision_inner_attention_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_rowwise_sharding_config,
)


_SIGLIP2_KEY_REPLACEMENTS = {
    "embeddings.patch_embedding.": "patch_embedding.",
    "embeddings.position_embedding.": "position_embedding.",
    "encoder.layers.": "layers.",
    ".layer_norm1.": ".norm1.",
    ".layer_norm2.": ".norm2.",
    ".self_attn.q_proj.": ".attn.wq.",
    ".self_attn.k_proj.": ".attn.wk.",
    ".self_attn.v_proj.": ".attn.wv.",
    ".self_attn.out_proj.": ".attn.proj.",
    ".mlp.fc1.": ".mlp.linear_fc1.",
    ".mlp.fc2.": ".mlp.linear_fc2.",
}


class ECGPositionEmbedding(nn.Embedding, Module):
    """Learned ECG positions stay whole on each TP rank, unlike vocab embeddings."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)


class SigLEP(Module):
    """Normalize and encode each ECG independently, then pool its patch tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int
        num_hidden_layers: int
        num_attention_heads: int
        num_leads: int = 12
        segment_length: int = 2500
        patch_size: int = 25
        num_ecg_tokens: int = 100
        layer_norm_eps: float = 1e-6
        hidden_act: str = "gelu_pytorch_tanh"
        attention_dropout: float = 0.0

        def __post_init__(self):
            dimensions = (
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_leads,
                self.segment_length,
                self.patch_size,
                self.num_ecg_tokens,
            )
            if min(dimensions) < 1 or self.segment_length % self.patch_size:
                raise ValueError(
                    "Dimensions must be positive and segment_length divisible by patch_size"
                )
            if self.hidden_size % self.num_attention_heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            if self.hidden_act not in ("gelu", "gelu_pytorch_tanh"):
                raise ValueError("SigLEP supports gelu and gelu_pytorch_tanh")
            if self.attention_dropout != 0:
                raise ValueError(
                    "SigLEP's FlexAttention backend requires attention_dropout=0"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        tp = spmd_mesh_size("tp")
        if config.num_attention_heads % tp:
            raise ValueError(
                "Tensor parallel degree must divide SigLEP attention heads"
            )
        dim = config.hidden_size
        patch_dim = config.num_leads * config.patch_size
        num_patches = config.segment_length // config.patch_size
        self.patch_embedding = Linear.Config(
            in_features=patch_dim,
            out_features=dim,
            bias=True,
            param_init={
                "weight": partial(nn.init.trunc_normal_, std=patch_dim**-0.5),
                "bias": nn.init.zeros_,
            },
            sharding_config=vision_invariant_linear_sharding_config(),
        ).build()
        self.position_embedding = ECGPositionEmbedding.Config(
            num_embeddings=num_patches,
            embedding_dim=dim,
            param_init={"weight": partial(nn.init.normal_, std=dim**-0.5)},
            sharding_config=ShardingConfig(
                state_shardings={
                    "weight": SpmdType({"dp": spmd.R, "cp": spmd.I, "tp": spmd.I})
                }
            ),
        ).build()
        norm = LayerNorm.Config(
            normalized_shape=dim,
            eps=config.layer_norm_eps,
            sharding_config=invariant_norm_sharding_config(),
        )
        block = VisionTransformerBlock.Config(
            norm1=norm,
            norm2=norm,
            attn=VisionAttention.Config(
                dim=dim,
                num_heads=config.num_attention_heads,
                rope=None,
                wq=_colwise(dim, dim, input_tp=spmd.R),
                wk=_colwise(dim, dim, input_tp=spmd.R),
                wv=_colwise(dim, dim, input_tp=spmd.R),
                proj=_rowwise(dim, dim),
                inner_attention=FlexInnerAttention.Config(
                    sharding_config=vision_inner_attention_sharding_config()
                ),
                sharding_config=vision_attention_sharding_config(positions_dp=spmd.R),
            ),
            mlp=VisionMLP.Config(
                fc1=_colwise(dim, config.intermediate_size),
                fc2=_rowwise(config.intermediate_size, dim),
                act_fn=GELU.Config(
                    approximate=(
                        "tanh" if config.hidden_act == "gelu_pytorch_tanh" else "none"
                    )
                ),
            ),
        )
        self.layers = ModuleDict(
            {str(i): block.build() for i in range(config.num_hidden_layers)}
        )
        self.post_layernorm = norm.build()

    def load_siglip2_hf(self, checkpoint_dir: str, *, ecg_embeddings: bool) -> None:
        """Load an HF vision tower, including ECG embeddings for SigLEP."""
        state = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("projector.")
            and (ecg_embeddings or key.startswith(("layers.", "post_layernorm.")))
        }
        prefix = "model." if ecg_embeddings else "vision_model."
        hf_state = {}
        for key, value in state.items():
            for hf_name, native_name in reversed(_SIGLIP2_KEY_REPLACEMENTS.items()):
                key = key.replace(native_name, hf_name)
            hf_state[f"{prefix}{key}"] = value
        dcp.load(hf_state, storage_reader=dcp.HuggingFaceStorageReader(checkpoint_dir))
        self.load_state_dict(state, strict=False)

    def load_siglip2_checkpoint(self, checkpoint_path: str) -> None:
        """Load a saved ECG SigLIP2 tower, including its ECG embeddings.

        Expects the wrapper's model_state_dict with vision_encoder.vision_model
        keys. The contrastive pooling head and text tower are unused by Orah;
        its projector keeps its initialized weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prefix = "vision_encoder.vision_model."
        converted = {}
        for key, value in checkpoint["model_state_dict"].items():
            if not key.startswith(prefix):
                continue
            key = key.removeprefix(prefix)
            if key.startswith("head."):
                continue
            for source, target in _SIGLIP2_KEY_REPLACEMENTS.items():
                key = key.replace(source, target)
            converted[key] = value

        expected = {
            key for key in self.state_dict() if not key.startswith("projector.")
        }
        missing = expected - converted.keys()
        unexpected = converted.keys() - expected
        if missing or unexpected:
            raise ValueError(
                "SigLIP2 ECG tower does not match SigLEP: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        set_model_state_dict(
            self,
            converted,
            options=StateDictOptions(full_state_dict=True, strict=False),
        )

    def forward(self, ecg_values: torch.Tensor) -> torch.Tensor:
        config = self.config
        if ecg_values.ndim != 3 or tuple(ecg_values.shape[1:]) != (
            config.num_leads,
            config.segment_length,
        ):
            raise ValueError(
                f"Expected ECG shape (batch, {config.num_leads}, {config.segment_length}), "
                f"got {tuple(ecg_values.shape)}"
            )
        ecg_BCL = ecg_values.to(self.patch_embedding.weight)
        low_B11 = ecg_BCL.amin((-2, -1), keepdim=True)
        ecg_BCL = (ecg_BCL - low_B11) / (
            ecg_BCL.amax((-2, -1), keepdim=True) - low_B11 + 1e-6
        )
        patches_BPK = (
            ecg_BCL.unfold(-1, config.patch_size, config.patch_size)
            .transpose(1, 2)
            .flatten(2)
        )
        batch, num_patches = patches_BPK.shape[:2]
        positions_P = torch.arange(num_patches, device=ecg_BCL.device)
        if spmd.is_type_checking():
            positions_P = spmd.mutate_type(positions_P, "tp", src=spmd.R, dst=spmd.I)
            positions_P = spmd.mutate_type(positions_P, "cp", src=spmd.R, dst=spmd.I)
        x_BPD = self.patch_embedding(patches_BPK.flatten(0, 1)).view(
            batch, num_patches, -1
        )
        x_TD = (x_BPD + self.position_embedding(positions_P)[None]).flatten(0, 1)
        with spmd.no_typecheck():
            attention_mask = create_block_diagonal_mask(
                torch.full((batch,), num_patches, device=x_TD.device, dtype=torch.long),
                batch * num_patches,
                x_TD.device,
            )
        for layer in self.layers.values():
            x_TD = layer(
                x_TD,
                positions=positions_P,
                attention_mask=attention_mask,
            )
        x_BPD = self.post_layernorm(x_TD).view(batch, num_patches, -1)
        return F.adaptive_avg_pool1d(
            x_BPD.transpose(1, 2), config.num_ecg_tokens
        ).transpose(1, 2)


class ProjectedSigLEP(SigLEP):
    """SigLEP and Orah's projector in one first-stage FSDP unit."""

    @dataclass(kw_only=True, slots=True)
    class Config(SigLEP.Config):
        output_size: int

        def build_flops_estimator(self, encoder: "ProjectedSigLEP") -> FlopsEstimator:
            num_patches = self.segment_length // self.patch_size
            patch_flops = active_parameter_flops_per_token(
                encoder, excluded_submodules=(encoder.projector,)
            )
            attention_flops = (
                12 * self.hidden_size * self.num_hidden_layers * num_patches**2
            )
            projector_flops = active_parameter_flops_per_token(encoder.projector)
            flops_per_ecg = (
                num_patches * patch_flops
                + attention_flops
                + self.num_ecg_tokens * projector_flops
            )

            def estimate(input_dict) -> int:
                ecgs = input_dict.get("ecg_values")
                return 0 if ecgs is None else ecgs.shape[0] * flops_per_ecg

            return estimate

    def __init__(self, config: Config):
        super().__init__(config)
        self.projector = VisionMLP.Config(
            fc1=_colwise(config.hidden_size, config.output_size),
            fc2=_rowwise(config.output_size, config.output_size),
            act_fn=GELU.Config(approximate="none"),
        ).build()

    @disable_dense_sp()
    def forward(self, ecg_values: torch.Tensor) -> torch.Tensor:
        features_BND = super().forward(ecg_values)
        return self.projector(features_BND.flatten(0, 1))


def _colwise(dim, output_dim, *, input_tp=spmd.I):
    linear_cls = Linear if input_tp == spmd.R else ColumnParallelLinear
    return linear_cls.Config(
        in_features=dim,
        out_features=output_dim,
        bias=True,
        sharding_config=vision_colwise_sharding_config(input_tp=input_tp),
    )


def _rowwise(dim, output_dim):
    return RowParallelLinear.Config(
        in_features=dim,
        out_features=output_dim,
        bias=True,
        sharding_config=vision_rowwise_sharding_config(),
    )


def projected_encoder_config(
    config: SigLEP.Config, *, output_size: int
) -> ProjectedSigLEP.Config:
    return ProjectedSigLEP.Config(
        **{
            f.name: getattr(config, f.name)
            for f in fields(SigLEP.Config)
            if f.name != "sharding_config"
        },
        output_size=output_size,
    )
