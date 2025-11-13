#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from accelerate.optimizer import AcceleratedOptimizer
except ImportError:  # pragma: no cover
    AcceleratedOptimizer = None

if AcceleratedOptimizer is not None:

    def _safe_train(self):
        inner = getattr(self.optimizer, "train", None)
        if callable(inner):
            return inner()
        return None

    def _safe_eval(self):
        inner = getattr(self.optimizer, "eval", None)
        if callable(inner):
            return inner()
        return None

    AcceleratedOptimizer.train = _safe_train  # type: ignore[assignment]
    AcceleratedOptimizer.eval = _safe_eval  # type: ignore[assignment]


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"cannot interpret boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune Gemma-3-1B-IT on FinQA.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--train_file", type=str, default="FinQA/dataset/train.json")
    parser.add_argument("--eval_file", type=str, default="FinQA/dataset/dev.json")
    parser.add_argument("--output_dir", type=str, default="__output__")
    parser.add_argument("--input_mode", type=str, choices={"gold", "all"}, default="gold")
    parser.add_argument(
        "--target_field",
        type=str,
        choices={"program", "exe_ans", "answer"},
        default="program",
    )
    parser.add_argument("--task_instruction", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices={"eager", "sdpa", "flash_attention_2"},
        default="eager",
    )
    parser.add_argument(
        "--eval_strategy",
        "--evaluation_strategy",
        dest="eval_strategy",
        type=str,
        choices={"no", "steps", "epoch"},
        default="steps",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices={"steps", "epoch"},
        default="steps",
    )
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    return parser.parse_args()


def table_to_lines(table: Sequence[Sequence[str]]) -> List[str]:
    lines: List[str] = []
    for row in table or []:
        cleaned = [cell.strip() for cell in row if cell and cell.strip()]
        if cleaned:
            lines.append(" | ".join(cleaned))
    return lines


def build_context(entry: Dict, mode: str) -> str:
    qa = entry["qa"]
    if mode == "gold":
        gold = qa.get("gold_inds") or {}
        ordered = [gold[key].strip() for key in sorted(gold)]
        if ordered:
            return "\n".join(ordered)
    parts: List[str] = []
    pre = entry.get("pre_text") or []
    post = entry.get("post_text") or []
    table = table_to_lines(entry.get("table"))
    if pre:
        parts.append("PRE TEXT:\n" + "\n".join(pre))
    if table:
        parts.append("TABLE:\n" + "\n".join(table))
    if post:
        parts.append("POST TEXT:\n" + "\n".join(post))
    return "\n\n".join(parts) if parts else "No additional context provided."


def build_prompt(question: str, context: str, hint: str) -> str:
    return (
        f"{hint.strip()}\n\nQuestion:\n{question.strip()}\n\nContext:\n{context.strip()}\n\nAnswer:"
    )


def extract_target(qa: Dict, field: str) -> str:
    target = qa.get(field, "")
    if isinstance(target, list):
        target = " ".join(target)
    if not target:
        fallback = qa.get("exe_ans") or qa.get("answer") or "unknown"
        target = fallback
    return str(target).strip()


def encode_example(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target: str,
    max_length: int,
) -> Dict[str, List[int]]:
    target_text = (target or "unknown").strip()
    eos = tokenizer.eos_token or ""
    target_with_eos = f"{target_text}{eos}"
    target_ids = tokenizer(
        target_with_eos,
        add_special_tokens=False,
    ).input_ids
    if len(target_ids) >= max_length:
        target_ids = target_ids[-max_length:]
        prompt_ids: List[int] = []
    else:
        remaining = max_length - len(target_ids)
        prompt_ids = (
            tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=remaining,
            ).input_ids
            if remaining > 0
            else []
        )
    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class FinQADataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        input_mode: str,
        target_field: str,
        task_instruction: str,
        max_samples: int = 0,
    ):
        raw_data = json.loads(Path(path).read_text())
        if max_samples:
            raw_data = raw_data[:max_samples]
        hint = task_instruction.strip()
        if not hint:
            if target_field == "program":
                hint = "Generate the FinQA reasoning program (DSL) that answers the question."
            else:
                hint = "Return the final numeric answer."
        self.features = [
            encode_example(
                tokenizer=tokenizer,
                prompt=build_prompt(
                    entry["qa"]["question"],
                    build_context(entry, input_mode),
                    hint,
                ),
                target=extract_target(entry["qa"], target_field),
                max_length=max_length,
            )
            for entry in raw_data
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


@dataclass
class LMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for item in features:
            diff = max_length - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [pad_id] * diff)
            batch_attention.append(item["attention_mask"] + [0] * diff)
            batch_labels.append(item["labels"] + [self.label_pad_token_id] * diff)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def slugify(value: str) -> str:
    cleaned = value.replace(os.sep, "-").replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    result = "".join(ch if ch in allowed else "-" for ch in cleaned)
    result = result.strip("-_.")
    return result or "run"


def build_experiment_dir(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    label = slugify(Path(args.model_name).name)
    lr_token = slugify(f"{args.learning_rate:.1e}")
    parts = [
        label,
        f"t{slugify(args.target_field)}",
        f"i{slugify(args.input_mode)}",
        f"seq{args.max_seq_length}",
        f"bs{args.per_device_train_batch_size}",
        f"ga{args.gradient_accumulation_steps}",
        f"lr{lr_token}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    ]
    name = "_".join(parts)
    experiment_dir = base / name
    suffix = 1
    while experiment_dir.exists():
        suffix += 1
        experiment_dir = base / f"{name}--{suffix}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


class FileLoggerCallback(TrainerCallback):
    def __init__(
        self,
        metrics_file: Path,
        train_loss_file: Path,
        eval_loss_file: Path,
    ):
        self.metrics_file = metrics_file
        self.train_loss_file = train_loss_file
        self.eval_loss_file = eval_loss_file
        for target in (self.metrics_file, self.train_loss_file, self.eval_loss_file):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)

    def _write_metrics(self, record: Dict):
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_plain(self, path: Path, step: int, epoch: float, value_name: str, value):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{step}\t{epoch}\t{value_name}\t{value}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {
            "type": "train",
            "step": state.global_step,
            "epoch": state.epoch,
        }
        record.update({k: self._to_serializable(v) for k, v in logs.items()})
        self._write_metrics(record)
        if "loss" in logs:
            self._write_plain(
                self.train_loss_file,
                state.global_step,
                state.epoch if state.epoch is not None else 0,
                "loss",
                logs["loss"],
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        record = {
            "type": "eval",
            "step": state.global_step,
            "epoch": state.epoch,
        }
        record.update({k: self._to_serializable(v) for k, v in metrics.items()})
        self._write_metrics(record)
        if "eval_loss" in metrics:
            self._write_plain(
                self.eval_loss_file,
                state.global_step,
                state.epoch if state.epoch is not None else 0,
                "eval_loss",
                metrics["eval_loss"],
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    experiment_dir = build_experiment_dir(args)
    args_dict = vars(args).copy()
    args_dict["resolved_output_dir"] = str(experiment_dir)
    config_path = experiment_dir / "args.json"
    config_path.write_text(json.dumps(args_dict, indent=2, sort_keys=True))
    metrics_file = experiment_dir / "metrics.jsonl"
    train_loss_file = experiment_dir / "train_loss.log"
    eval_loss_file = experiment_dir / "eval_loss.log"

    print(f"Experiment directory: {experiment_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = FinQADataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        input_mode=args.input_mode,
        target_field=args.target_field,
        task_instruction=args.task_instruction,
        max_samples=args.max_train_samples,
    )

    eval_dataset = (
        FinQADataset(
            path=args.eval_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            input_mode=args.input_mode,
            target_field=args.target_field,
            task_instruction=args.task_instruction,
            max_samples=args.max_eval_samples,
        )
        if args.do_eval
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, attn_implementation=args.attn_implementation
    )
    model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(experiment_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy if args.do_eval else "no",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        load_best_model_at_end=args.load_best_model_at_end and args.do_eval,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LMDataCollator(tokenizer),
        callbacks=[
            FileLoggerCallback(
                metrics_file=metrics_file,
                train_loss_file=train_loss_file,
                eval_loss_file=eval_loss_file,
            )
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(experiment_dir)

    if args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        print(metrics)


if __name__ == "__main__":
    main()
