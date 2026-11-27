#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export ORAH_STAGE=2
export ORAH_CHECKPOINT="${1:?Pass the SFT stage 1 checkpoint directory}"
shift
NGPU="${NGPU:-6}" MODULE=torchtitan_recipes.sft.orah CONFIG=orah_sft bash run_train.sh "$@"
