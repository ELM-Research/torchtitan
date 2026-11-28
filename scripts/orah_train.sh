#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export NGPU="${NGPU:-8}"
siglep_checkpoint="${ORAH_SIGLEP_CHECKPOINT:-/p01/whan/ecg_encoder/src/runs/pretrain/siglip2-base-patch16-naflex/2/checkpoints/epoch_best.pt}"
unset ORAH_BENCHMARK_STEPS ORAH_PIPELINE_DEBUG ORAH_CHECKPOINT ORAH_SIGLEP_CHECKPOINT
mkdir -p outputs/orah_train
run_dir=$(mktemp -d outputs/orah_train/run.XXXXXXXX)
printf 'Saving training outputs under %s\n' "$run_dir"

ORAH_STAGE=1 ORAH_SIGLEP_CHECKPOINT="$siglep_checkpoint" \
    MODULE=torchtitan_recipes.orah CONFIG=orah_pretrain \
    bash run_train.sh --output-dir "$run_dir/pretrain1"

checkpoint=$(ls -d "$run_dir/pretrain1"/checkpoint/step-*/.metadata | sort -V | tail -n 1)
ORAH_STAGE=2 ORAH_CHECKPOINT="${checkpoint%/.metadata}" \
    MODULE=torchtitan_recipes.orah CONFIG=orah_pretrain \
    bash run_train.sh --output-dir "$run_dir/pretrain2"

checkpoint=$(ls -d "$run_dir/pretrain2"/checkpoint/step-*/.metadata | sort -V | tail -n 1)
ORAH_STAGE=1 ORAH_CHECKPOINT="${checkpoint%/.metadata}" \
    MODULE=torchtitan_recipes.sft.orah CONFIG=orah_sft \
    bash run_train.sh --output-dir "$run_dir/sft1"

checkpoint=$(ls -d "$run_dir/sft1"/checkpoint/step-*/.metadata | sort -V | tail -n 1)
ORAH_STAGE=2 ORAH_CHECKPOINT="${checkpoint%/.metadata}" \
    MODULE=torchtitan_recipes.sft.orah CONFIG=orah_sft \
    bash run_train.sh --output-dir "$run_dir/sft2"

ls -d "$run_dir/sft2"/checkpoint/step-*/.metadata >/dev/null
