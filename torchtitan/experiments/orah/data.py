# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import grain.python as grain
import numpy as np
import torch
from datasets import Dataset, load_dataset

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer


ECG_TOKEN = "<ecg>"
END_OF_TEXT = "<|endoftext|>"
_SYSTEM_PROMPT = """You are an expert cardiac electrophysiologist. You interpret 12-lead ECG signals and answer any questions about them.

Deliver responses that are precise and grounded in electrophysiology and clinical reasoning.
For general, non-ECG questions, respond appropriately.

If the response opens with <think>, close it with </think>, then give a brief interpretation if necessary and end with <answer>...</answer>.
Otherwise answer the question directly, with no tags.
Never repeat a tag that has already been written."""
_MODALITY_TAG = re.compile(r"<\s*(?:ecg|image)\s*>\s*", re.IGNORECASE)
_IMAGE_WORD = re.compile(r"\b(image|picture)\b", re.IGNORECASE)
_ROLE_PREFIX = re.compile(
    r"^\s*(?:(?:user|assistant|human|gpt|model|system|q|a)\s*[:\uff1a]\s*|[:\uff1a]\s*)+",
    re.IGNORECASE,
)
_ASSISTANT_RESPONSE = re.compile(
    r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL
)


def load_ecg(path: Path) -> np.ndarray:
    signal = np.load(path, allow_pickle=True)
    return np.asarray(signal.item()["ecg"], dtype=np.float32)


def prepare_sft_messages(messages_from_row, *, num_ecg_tokens: int) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for message in messages_from_row:
        content = _MODALITY_TAG.sub("", message["content"])
        content = _IMAGE_WORD.sub(
            lambda match: "Signal" if match[0][0].isupper() else "signal", content
        )
        messages.append(
            {"role": message["role"], "content": _ROLE_PREFIX.sub("", content)}
        )
    user = next(message for message in messages if message["role"] == "user")
    user["content"] = ECG_TOKEN * num_ecg_tokens + "\n" + user["content"]
    return messages


class OrahTokenizer(HuggingFaceTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def __init__(self, config: Config, *, tokenizer_path: str):
        super().__init__(config, tokenizer_path=tokenizer_path)
        self.tokenizer.add_tokens([ECG_TOKEN])


@dataclass(frozen=True, kw_only=True, slots=True)
class OrahDataset(ABC):
    stage: int | None
    num_rows_per_train_step_per_dp_rank: int
    ecg_root: str
    num_ecg_tokens: int

    @staticmethod
    @abstractmethod
    def rows(stage: int | None) -> Dataset: ...

    @abstractmethod
    def text_processor(self, context: DatasetBuildContext): ...

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        rows = self.rows(self.stage)
        prepare_text = self.text_processor(context)
        ecg_root = Path(self.ecg_root)

        def prepare(index):
            row = rows[index]
            text = prepare_text(row["text"])
            if row["ecg_path"] is None:
                ecg = np.random.default_rng(0).standard_normal(
                    (12, 2500), dtype=np.float32
                )
            else:
                ecg = load_ecg(ecg_root / row["ecg_path"].lstrip("/"))
            return text, ecg

        policy = dataset_iteration_policy
        global_batch_size = (
            self.num_rows_per_train_step_per_dp_rank * policy.dp_world_size
        )
        num_rows = len(rows) // global_batch_size * global_batch_size
        if num_rows == 0:
            raise ValueError("Orah needs at least one global batch")
        dataset = grain.MapDataset.range(len(rows))
        if policy.shuffle:
            dataset = dataset.shuffle(seed=policy.seed)
        # Keep every optimizer step within one epoch, including with accumulation.
        dataset = dataset[policy.dp_rank : num_rows : policy.dp_world_size]
        if policy.repeat:
            dataset = dataset.repeat()
        return dataset.map(prepare)


class OrahSFTDataset(OrahDataset):
    @staticmethod
    @cache
    def rows(stage: int | None) -> Dataset:
        if stage is not None:
            return load_dataset(f"ELM-Research/orah-sft-{stage}", split="train")
        return Dataset.from_list(
            [
                {
                    "ecg_path": None,
                    "text": [
                        {"role": "user", "content": "Can you see the ECG?"},
                        {"role": "assistant", "content": "Yes."},
                        {"role": "user", "content": "Interpret this ECG."},
                        {
                            "role": "assistant",
                            "content": "<ecg>\n<think>\nSynthetic signal.\n</think>\n\n<answer>Debug example</answer>",
                        },
                    ],
                }
            ]
        )

    def text_processor(self, context: DatasetBuildContext):
        tokenizer = context.tokenizer
        assert isinstance(tokenizer, OrahTokenizer)
        if tokenizer.token_to_id("<|im_end|>") is None:
            raise ValueError("Orah SFT requires a Qwen3.5 tokenizer")
        limit = context.max_context_length

        def prepare(messages_from_row):
            messages = prepare_sft_messages(
                messages_from_row, num_ecg_tokens=self.num_ecg_tokens
            )
            last_user = max(
                index
                for index, message in enumerate(messages)
                if message["role"] == "user"
            )
            # Leave annotated reasoning in historical assistant replies.
            for message in messages[:last_user]:
                if message["role"] == "assistant":
                    message["reasoning_content"] = ""
            conversation = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False
            ).removesuffix("\n")
            encoding = tokenizer.tokenizer.encode(
                conversation, add_special_tokens=False
            )
            tokens = np.asarray(encoding.ids[:limit], dtype=np.int64)
            starts, ends = np.asarray(encoding.offsets[:limit]).T
            response_mask = np.zeros(len(tokens), dtype=bool)
            for response in _ASSISTANT_RESPONSE.finditer(conversation):
                start, end = response.span(1)
                response_mask |= (starts >= start) & (ends <= end)
            labels = np.where(response_mask, tokens, IGNORE_INDEX)
            return TextSequence(
                input_ids=tokens[:-1],
                labels=labels[1:],
            )

        return prepare


class OrahPretrainDataset(OrahDataset):
    @staticmethod
    @cache
    def rows(stage: int | None) -> Dataset:
        if stage is not None:
            return load_dataset(f"ELM-Research/orah-pretraining-{stage}", split="train")
        return Dataset.from_list(
            [{"ecg_path": None, "text": "<ecg>\nsynthetic ECG report"}]
        )

    def text_processor(self, context: DatasetBuildContext):
        tokenizer = context.tokenizer
        assert isinstance(tokenizer, OrahTokenizer)
        separator = tokenizer.token_to_id(END_OF_TEXT)
        if separator is None:
            raise ValueError("Orah pretraining requires Qwen's <|endoftext|> token")
        prefix = tokenizer.encode(
            ECG_TOKEN * self.num_ecg_tokens + "\n", add_bos=False, add_eos=False
        )
        max_report_length = context.max_context_length - len(prefix) - 1
        if max_report_length < 1:
            raise ValueError("Orah pretraining context has no room for the report")

        def prepare(text):
            report = tokenizer.encode(
                text.removeprefix(f"{ECG_TOKEN}\n"),
                add_bos=False,
                add_eos=False,
            )[:max_report_length]
            # Supervise the separator but mask the ECG prefix and its newline.
            return TextSequence(
                input_ids=np.asarray(prefix + report, dtype=np.int64),
                labels=np.asarray(
                    [IGNORE_INDEX] * (len(prefix) - 1) + report + [separator],
                    dtype=np.int64,
                ),
            )

        return prepare


class OrahCollator(TextCollator):
    @dataclass(kw_only=True, slots=True)
    class Config(TextCollator.Config):
        batch_size: int
        pad_token_id: int = 0

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        super().__init__(config, context=context)
        self.batch_size = config.batch_size
        self.pad_token_id = config.pad_token_id

    def num_rows_per_microbatch(self):
        return self.batch_size

    def __call__(self, rows):
        texts, ecgs = zip(*rows, strict=True)
        microbatch = super().__call__(texts)
        if self.pad_token_id:
            microbatch.input.masked_fill_(microbatch.padding_mask, self.pad_token_id)
        microbatch.model_kwargs["ecg_values"] = torch.from_numpy(np.stack(ecgs))
        return microbatch
