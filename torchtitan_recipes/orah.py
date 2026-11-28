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
from torchtitan.experiments.orah.data import (
    ECG_TOKEN,
    END_OF_TEXT,
    OrahCollator,
    OrahDataset,
    OrahPretrainDataset,
    OrahTokenizer,
)
from torchtitan.trainer import Trainer


def orah_training(
    stage: int | None,
    *,
    name: str,
    dataset: type[OrahDataset],
    lr: float,
    epochs: int,
    trainable: tuple[str, ...] = ("projector", "language_model"),
    initial_checkpoint: str | None = None,
    siglep_checkpoint: str | None = None,
    pad_token: str | None = None,
) -> Trainer.Config:
    """Build an Orah recipe.

    For real training, ORAH_MICROBATCH_SIZE sets rows per rank per microbatch,
    ORAH_SEQ_LEN sets the context cap, and ORAH_GRAD_ACCUM_STEPS sets the number
    of microbatches per optimizer step. ORAH_BENCHMARK_STEPS runs that many
    optimizer steps on real data. The pipeline debug launcher sets
    ORAH_PIPELINE_DEBUG=1 to save a final checkpoint for the next stage.
    """
    debug = stage is None
    model_size = os.environ.get("ORAH_MODEL", "4B")
    if model_size not in ("4B", "9B"):
        raise ValueError("ORAH_MODEL must be 4B or 9B")
    flavor = "debugmodel" if debug else model_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not debug and world_size < 2:
        raise ValueError(
            "Orah training uses DistMuon with FSDP and requires at least two GPUs"
        )

    batch_size = {"debugmodel": 1, "4B": 12, "9B": 10}[flavor]
    seq_len = 384 if debug else 1024
    gradient_accumulation_steps = 1
    benchmark_steps = None
    pipeline_debug = not debug and os.environ.get("ORAH_PIPELINE_DEBUG") == "1"
    if not debug:
        batch_size = int(os.environ.get("ORAH_MICROBATCH_SIZE", batch_size))
        seq_len = int(os.environ.get("ORAH_SEQ_LEN", seq_len))
        gradient_accumulation_steps = int(os.environ.get("ORAH_GRAD_ACCUM_STEPS", "1"))
        if batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError(
                "ORAH_MICROBATCH_SIZE and ORAH_GRAD_ACCUM_STEPS must be positive"
            )
        if seq_len <= 0 or seq_len % 2:
            raise ValueError("ORAH_SEQ_LEN must be positive and even for chunked loss")
        if "ORAH_BENCHMARK_STEPS" in os.environ:
            benchmark_steps = int(os.environ["ORAH_BENCHMARK_STEPS"])
            if benchmark_steps < 1:
                raise ValueError("ORAH_BENCHMARK_STEPS must be positive")
            if not pipeline_debug and benchmark_steps < 10:
                raise ValueError("ORAH_BENCHMARK_STEPS must be at least 10")
        elif pipeline_debug:
            raise ValueError("ORAH_PIPELINE_DEBUG requires ORAH_BENCHMARK_STEPS")

    num_tokens_per_microbatch = batch_size * seq_len
    num_loss_chunks = batch_size * 2
    num_rows_per_train_step_per_dp_rank = batch_size * gradient_accumulation_steps

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
    pad_token_id = 0 if pad_token is None else tokenizer.token_to_id(pad_token)
    if pad_token_id is None:
        raise ValueError(f"Orah tokenizer is missing {pad_token}")
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
        trainable=trainable,
    )

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
            trainable_parameters = {
                name: param.ndim
                for name, param in model.build().named_parameters()
                if param.requires_grad
            }
        muon_parameters = [
            name
            for name, ndim in trainable_parameters.items()
            if (ndim == 2 and name not in ("tok_embeddings.weight", "lm_head.weight"))
            or name.endswith(".feed_forward.w13.weight")
        ]
        adamw_decay_parameters = [
            name
            for name, ndim in trainable_parameters.items()
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
        ]
        if adamw_decay_parameters:
            optimizers.append(
                AdamW.Config(
                    pattern=r"^(?:"
                    + "|".join(map(re.escape, adamw_decay_parameters))
                    + r")$",
                    lr=lr,
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=0.01,
                )
            )
        optimizers.append(adamw)
        steps_per_epoch = len(dataset.rows(stage)) // (
            num_rows_per_train_step_per_dp_rank * world_size
        )
        if steps_per_epoch == 0:
            raise ValueError("Orah needs at least one global batch")
        steps = benchmark_steps or epochs * steps_per_epoch
        checkpointer = OrahCheckpointer.Config(
            qwen_model=qwen_model,
            siglep_checkpoint=siglep_checkpoint,
            initial_load_path=(
                str(Path(initial_checkpoint).resolve()) if initial_checkpoint else None
            ),
            initial_load_model_only=initial_checkpoint is not None,
            interval=steps if pipeline_debug else steps_per_epoch,
            last_save_model_only=False,
            load_only=benchmark_steps is not None and not pipeline_debug,
        )

    if pipeline_debug:
        name = f"{name}_debug"
    elif benchmark_steps is not None:
        name = f"{name}_benchmark"
    torch.set_float32_matmul_precision("high")
    return Trainer.Config(
        model=model,
        hf_assets_path=tokenizer_path,
        tokenizer=tokenizer_config,
        parallelism=parallelism,
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            num_tokens_per_train_step=(
                num_tokens_per_microbatch * world_size * gradient_accumulation_steps
            ),
            steps=steps,
            disable_accelerator_graphs=True,
        ),
        activation_remat=None if debug else FullActivationRematConfig(),
        loss=CrossEntropyLoss.Config()
        if debug
        else ChunkedLossWrapper.Config(num_chunks=num_loss_chunks),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(optimizers=optimizers),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0 if debug else math.ceil(0.03 * steps),
                decay_type="cosine",
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=dataset(
                stage=stage,
                num_rows_per_train_step_per_dp_rank=num_rows_per_train_step_per_dp_rank,
                ecg_root=os.environ.get("ORAH_ECG_ROOT", "/"),
                num_ecg_tokens=encoder.num_ecg_tokens,
            ),
            collator=OrahCollator.Config(
                batch_size=batch_size,
                pad_to_multiple_of=1 if debug else num_loss_chunks,
                pad_token_id=pad_token_id,
            ),
            max_num_documents=batch_size,
            seed=0,
            read_options=grain.ReadOptions(num_threads=8),
        ),
        checkpointer=checkpointer,
        debug=DebugConfig(seed=0),
        dump_folder=f"./outputs/orah_{flavor.lower()}_{name}",
    )


def orah_pretrain() -> Trainer.Config:
    stage = int(os.environ["ORAH_STAGE"])
    if stage not in (1, 2):
        raise ValueError("ORAH_STAGE must be 1 or 2")
    initial_checkpoint = os.environ.get("ORAH_CHECKPOINT") or None
    if stage == 2 and initial_checkpoint is None:
        raise ValueError("Orah pretraining stage 2 requires a stage 1 checkpoint")
    siglep_checkpoint = os.environ.get("ORAH_SIGLEP_CHECKPOINT") or (
        "/p01/whan/ecg_encoder/src/runs/pretrain/siglip2-base-patch16-naflex/2/hf_converted"
    )
    return orah_training(
        stage,
        name=f"pretrain{stage}",
        dataset=OrahPretrainDataset,
        lr=5e-4 if stage == 1 else 2e-4,
        epochs=3 if stage == 1 else 1,
        trainable=("projector",) if stage == 1 else ("projector", "language_model"),
        initial_checkpoint=initial_checkpoint if stage == 2 else None,
        siglep_checkpoint=siglep_checkpoint if stage == 1 else None,
        pad_token=END_OF_TEXT,
    )


def orah_pretrain_debug() -> Trainer.Config:
    return orah_training(
        None,
        name="pretrain_debug",
        dataset=OrahPretrainDataset,
        lr=5e-4,
        epochs=1,
        trainable=("projector",),
        pad_token=END_OF_TEXT,
    )
