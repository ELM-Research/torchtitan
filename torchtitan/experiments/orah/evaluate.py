# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate an Orah TorchTitan checkpoint on the ECG benchmark test splits."""

# Shape suffixes: B = ECGs, C = leads, L = samples, T = text tokens,
# D = decoder width, V = vocabulary.

import argparse
import json
import os
import secrets
import string
from collections import Counter
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from torchtitan.config.configs import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.fsdp import maybe_store_frozen_params_in_param_dtype_
from torchtitan.experiments.orah import build_model_config, SigLEP
from torchtitan.experiments.orah.data import (
    ECG_TOKEN,
    load_ecg,
    OrahTokenizer,
    prepare_sft_messages,
)
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
)


_BENCHMARKS = {"r1": "ECG-R1-Benchmark", "cot": "ECG-QA-CoT-Benchmark"}
_REMOVE_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _split_response(text: str) -> tuple[str, str]:
    thinking = ""
    in_thinking = "<think>" in text
    if in_thinking:
        text = text.partition("<think>")[2]
    if in_thinking or "</think>" in text:
        thinking, closed, text = text.partition("</think>")
        if not closed:
            return thinking.strip(), ""
    if "<answer>" in text:
        text = text.partition("<answer>")[2]
    return thinking.strip(), text.partition("</answer>")[0].strip()


def _token_f1(reference: str, prediction: str) -> float:
    reference_tokens = reference.lower().translate(_REMOVE_PUNCTUATION).split()
    prediction_tokens = prediction.lower().translate(_REMOVE_PUNCTUATION).split()
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    return 2 * overlap / (len(reference_tokens) + len(prediction_tokens))


def _metrics(records: list[dict]) -> dict:
    answers, thinking = [], []
    for record in records:
        reference_thinking, reference_answer = _split_response(record["ground_truth"])
        predicted_thinking, predicted_answer = _split_response(record["predicted"])
        answers.append((reference_answer, predicted_answer))
        if reference_thinking:
            thinking.append((reference_thinking, predicted_thinking))

    return {
        name: {
            "accuracy": sum(ref == pred for ref, pred in pairs) / len(pairs),
            "f1": sum(_token_f1(ref, pred) for ref, pred in pairs) / len(pairs),
        }
        for name, pairs in (("answer", answers), ("thinking", thinking))
        if pairs
    }


def _load_model(checkpoint, *, flavor, tokenizer, device, max_new_tokens):
    config = build_model_config(
        flavor,
        encoder=SigLEP.Config(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
        ),
        ecg_token_id=tokenizer.token_to_id(ECG_TOKEN),
        parallelism=ParallelismConfig(),
        seq_len=4096 + max_new_tokens,
        attn_backend="flex",
        vocab_size=tokenizer.get_vocab_size(),
    )
    with torch.device("meta"):
        model = config.build()
    model.requires_grad_(False)
    maybe_store_frozen_params_in_param_dtype_(model, PrecisionConfig())
    model.to_empty(device=device)
    model.init_states(buffer_device=device)
    state = model.state_dict()
    dcp.load(state, checkpoint_id=checkpoint)
    model.load_state_dict(state)
    print(f"Loaded Orah checkpoint from {checkpoint}", flush=True)
    model._skip_lm_head = True
    return model.eval()


@torch.inference_mode()
def _generate(model, *, tokenizer, prompt, ecg_path, device, max_new_tokens):
    prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    if len(prompt_ids) + max_new_tokens > model.config.max_context_length:
        raise ValueError(f"Prompt for {ecg_path} exceeds the evaluation context")
    if (
        prompt_ids.count(model.config.ecg_token_id)
        != model.config.vision_encoder.num_ecg_tokens
    ):
        raise ValueError(f"Incorrect number of ECG placeholders for {ecg_path}")
    ecg_BCL = torch.as_tensor(load_ecg(ecg_path), device=device).unsqueeze(0)
    tokens_T = torch.tensor(prompt_ids, device=device)
    inputs_TD = gather_vision_embeds(
        model.tok_embeddings(tokens_T),
        vision_bank_VD=model.vision_encoder(ecg_BCL),
        vision_bank_indices_T=build_vision_bank_indices(
            tokens_T, placeholder_id=model.config.ecg_token_id
        ),
    )
    stop_ids = {tokenizer.eos_id, tokenizer.token_to_id("<|im_end|>")}
    positions_T = torch.arange(len(prompt_ids) + max_new_tokens, device=device)
    generated = []
    for _ in range(max_new_tokens):
        step_positions_T = positions_T[: len(inputs_TD)]
        hidden_TD = model.forward_embeddings(
            inputs_TD,
            step_positions_T,
            model._get_attention_metadata(step_positions_T),
        )
        logits_V = model.lm_head(hidden_TD[-1])
        next_id = logits_V.argmax()
        token = next_id.item()
        if token in stop_ids:
            break
        generated.append(token)
        inputs_TD = torch.cat((inputs_TD, model.tok_embeddings(next_id[None])))
    return tokenizer.decode(generated, skip_special_tokens=False).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="TorchTitan checkpoint/step-N")
    parser.add_argument("--benchmark", choices=_BENCHMARKS, help="Both by default")
    parser.add_argument(
        "--model", choices=("4B", "9B"), default=os.getenv("ORAH_MODEL", "4B")
    )
    parser.add_argument("--seed", type=int, help="A fresh random seed by default")
    parser.add_argument("--limit", type=int, help="Evaluate N randomly selected rows")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not (args.checkpoint / ".metadata").is_file():
        parser.error(f"Expected a TorchTitan DCP step directory: {args.checkpoint}")
    if args.limit is not None and (args.limit < 1 or args.benchmark is None):
        parser.error("--limit requires a positive count and --benchmark")
    if args.seed is not None and not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        parser.error("Orah evaluation requires a CUDA GPU")

    seed = secrets.randbits(32) if args.seed is None else args.seed
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    tokenizer_path = os.environ.get("ORAH_TOKENIZER") or snapshot_download(
        f"Qwen/Qwen3.5-{args.model}-Base",
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    tokenizer = OrahTokenizer.Config().build(tokenizer_path=tokenizer_path)
    model = _load_model(
        args.checkpoint,
        flavor=args.model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    ecg_root = Path(os.environ.get("ORAH_ECG_ROOT", "/"))
    benchmarks = (
        [_BENCHMARKS[args.benchmark]] if args.benchmark else _BENCHMARKS.values()
    )
    for benchmark in benchmarks:
        dataset = load_dataset(f"ELM-Research/{benchmark}", split="test")
        run_dir = args.checkpoint
        if args.limit is not None:
            if args.limit > len(dataset):
                parser.error(f"--limit exceeds the {len(dataset)} rows in {benchmark}")
            dataset = dataset.shuffle(seed=seed).select(range(args.limit))
            run_dir /= f"debug_{args.limit}"
        print(f"Evaluating {benchmark} with seed {seed}", flush=True)
        records = []
        for row in tqdm(dataset, desc=benchmark):
            *history, reference = prepare_sft_messages(
                row["text"], num_ecg_tokens=model.config.vision_encoder.num_ecg_tokens
            )
            prompt = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, enable_thinking=True
            ).removesuffix("<think>\n")
            prediction = _generate(
                model,
                tokenizer=tokenizer,
                prompt=prompt,
                ecg_path=ecg_root / row["ecg_path"].lstrip("/"),
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            records.append(
                {
                    "prompt": prompt,
                    "ground_truth": reference["content"],
                    "predicted": prediction,
                }
            )

        run_dir /= f"ELM-Research_{benchmark}"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = run_dir / f"seed_{seed}.json"
        predictions.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        run = {
            "seed": seed,
            "num_pairs": len(records),
            "metrics": _metrics(records),
            "file": predictions.name,
        }
        summary = run_dir / "summary.json"
        summary.write_text(
            json.dumps({"runs": [run]}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(run, indent=2))
        print(f"Saved evaluation results to {summary}")


if __name__ == "__main__":
    main()
