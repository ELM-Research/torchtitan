# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torchtitan.experiments.orah.data import OrahSFTDataset
from torchtitan.trainer import Trainer
from torchtitan_recipes.orah import orah_training


def orah_sft() -> Trainer.Config:
    stage = int(os.environ["ORAH_STAGE"])
    if stage not in (1, 2):
        raise ValueError("ORAH_STAGE must be 1 or 2")
    return orah_training(
        stage,
        name=f"sft{stage}_conversations",
        dataset=OrahSFTDataset,
        lr=5e-5 if stage == 2 else 1e-4,
        epochs=3,
        initial_checkpoint=os.environ.get("ORAH_CHECKPOINT") or None,
        siglep_checkpoint=os.environ.get("ORAH_SIGLEP_CHECKPOINT") or None,
    )


def orah_sft_debug() -> Trainer.Config:
    return orah_training(None, name="debug", dataset=OrahSFTDataset, lr=1e-4, epochs=1)
