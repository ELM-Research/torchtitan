# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
from pathlib import Path

import grain.python as grain
import torch
from huggingface_hub import snapshot_download

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    BaseOptimizer,
    DistMuon,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config.configs import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.experiments.orah import build_model_config, SigLEP
from torchtitan.experiments.orah.checkpointer import OrahCheckpointer
from torchtitan.experiments.orah.sft_data import (
    ECG_TOKEN,
    OrahSFTCollator,
    OrahSFTDataset,
    OrahTokenizer,
    sft_rows,
)
from torchtitan.trainer import Trainer


def _orah_sft(stage: int | None) -> Trainer.Config:
    debug = stage is None
    model_size = os.environ.get("ORAH_MODEL", "4B")
    if model_size not in ("4B", "9B"):
        raise ValueError("ORAH_MODEL must be 4B or 9B")
    flavor = "debugmodel" if debug else model_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not debug and world_size < 2:
        raise ValueError(
            "Orah SFT uses DistMuon with FSDP and requires at least two GPUs"
        )

    qwen_model = f"Qwen/Qwen3.5-{model_size}-Base"
    tokenizer_path = os.environ.get("ORAH_TOKENIZER") or snapshot_download(
        qwen_model,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    tokenizer_config = OrahTokenizer.Config()
    tokenizer = tokenizer_config.build(tokenizer_path=tokenizer_path)
    ecg_token_id = tokenizer.token_to_id(ECG_TOKEN)
    assert ecg_token_id is not None
    batch_size = {"debugmodel": 1, "4B": 12, "9B": 10}[flavor]
    seq_len = 384 if debug else 1024
    encoder = SigLEP.Config(
        hidden_size=64 if debug else 768,
        intermediate_size=128 if debug else 3072,
        num_hidden_layers=2 if debug else 12,
        num_attention_heads=4 if debug else 12,
    )
    parallelism = ParallelismConfig(data_parallel_shard_degree=world_size)
    model = build_model_config(
        flavor,
        encoder=encoder,
        ecg_token_id=ecg_token_id,
        parallelism=parallelism,
        seq_len=seq_len,
        attn_backend="flex",
        vocab_size=tokenizer.get_vocab_size(),
    )

    lr = 5e-5 if stage == 2 else 1e-4
    adamw = AdamW.Config(
        pattern=r".*", lr=lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0
    )
    optimizers: list[BaseOptimizer.Config]
    if debug:
        optimizers = [adamw]
        steps = 1
        checkpointer = None
    else:
        with torch.device("meta"):
            trainable = {
                name: param.ndim
                for name, param in model.build().named_parameters()
                if param.requires_grad
            }
        muon_parameters = [
            name
            for name, ndim in trainable.items()
            if (ndim == 2 and name not in ("tok_embeddings.weight", "lm_head.weight"))
            or name.endswith(".feed_forward.w13.weight")
        ]
        adamw_decay_parameters = [
            name
            for name, ndim in trainable.items()
            if ndim > 1 and name not in muon_parameters
        ]
        # DistMuon updates TorchTitan's native matrices, including fused projections.
        owned = ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()},
            compute_dtype=torch.bfloat16,
        )
        optimizers = [
            DistMuon.Config(
                pattern=r"^(?:" + "|".join(map(re.escape, muon_parameters)) + r")$",
                compute_sharding_by_fqn=dict.fromkeys(muon_parameters, owned),
                bucket_configs=tuple(
                    BucketConfig(name=name, patterns=(name,))
                    for name in muon_parameters
                ),
                lr=lr,
                weight_decay=0.01,
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(
                pattern=r"^(?:"
                + "|".join(map(re.escape, adamw_decay_parameters))
                + r")$",
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=0.01,
            ),
            adamw,
        ]
        initial_checkpoint = os.environ.get("ORAH_CHECKPOINT") or None
        if initial_checkpoint is not None:
            initial_checkpoint = str(Path(initial_checkpoint).resolve())
        steps_per_epoch = len(sft_rows(stage)) // (batch_size * world_size)
        steps = 3 * steps_per_epoch
        checkpointer = OrahCheckpointer.Config(
            qwen_model=qwen_model,
            siglep_checkpoint=os.environ.get("ORAH_SIGLEP_CHECKPOINT") or None,
            initial_load_path=initial_checkpoint,
            initial_load_model_only=initial_checkpoint is not None,
            interval=steps_per_epoch,
            last_save_model_only=False,
        )

    torch.set_float32_matmul_precision("high")
    return Trainer.Config(
        model=model,
        hf_assets_path=tokenizer_path,
        tokenizer=tokenizer_config,
        parallelism=parallelism,
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=batch_size * seq_len,
            steps=steps,
            disable_accelerator_graphs=True,
        ),
        activation_remat=None if debug else FullActivationRematConfig(),
        loss=CrossEntropyLoss.Config()
        if debug
        else ChunkedLossWrapper.Config(num_chunks=batch_size * 2),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(optimizers=optimizers),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0 if debug else math.ceil(0.03 * steps),
                decay_type="cosine",
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=OrahSFTDataset(
                stage=stage,
                batch_size=batch_size,
                ecg_root=os.environ.get("ORAH_ECG_ROOT", "/"),
                num_ecg_tokens=encoder.num_ecg_tokens,
            ),
            collator=OrahSFTCollator.Config(batch_size=batch_size),
            max_num_documents=batch_size,
            seed=0,
            read_options=grain.ReadOptions(num_threads=8),
        ),
        checkpointer=checkpointer,
        debug=DebugConfig(seed=0),
        dump_folder=f"./outputs/orah_{flavor.lower()}_{'debug' if debug else f'sft{stage}_conversations'}",
    )


def orah_sft() -> Trainer.Config:
    stage = int(os.environ["ORAH_STAGE"])
    if stage not in (1, 2):
        raise ValueError("ORAH_STAGE must be 1 or 2")
    return _orah_sft(stage)


def orah_sft_debug() -> Trainer.Config:
    return _orah_sft(None)
