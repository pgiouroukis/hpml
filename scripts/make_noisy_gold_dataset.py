#!/usr/bin/env python3
"""
Utility to materialize FinQA prompts that include gold evidence plus random distractors.
Use this to precompute \"noisy gold\" training data for robustness experiments.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence


def table_to_lines(table: Sequence[Sequence[str]] | None) -> List[str]:
    if not table:
        return []
    lines: List[str] = []
    for row in table:
        cleaned = [cell.strip() for cell in row if cell and cell.strip()]
        if cleaned:
            lines.append(" | ".join(cleaned))
    return lines


def _sample_with_noise(
    lines: Sequence[str],
    gold_indices: Sequence[int],
    noise_k: int,
    rng: random.Random,
) -> List[str]:
    if not lines:
        return []
    gold_set = set(gold_indices)
    pool = [idx for idx in range(len(lines)) if idx not in gold_set]
    noise = rng.sample(pool, min(noise_k, len(pool))) if noise_k > 0 else []
    keep = sorted(gold_set.union(noise))
    return [lines[idx] for idx in keep]


def build_context(
    entry: Dict,
    mode: str,
    noisy_text_distractors: int,
    noisy_table_distractors: int,
    noisy_seed: int,
    example_index: int,
) -> str:
    qa = entry["qa"]
    if mode == "gold":
        gold = qa.get("gold_inds") or {}
        ordered = [gold[key].strip() for key in sorted(gold)]
        if ordered:
            return "\n".join(ordered)

    pre = entry.get("pre_text") or []
    post = entry.get("post_text") or []
    table_lines = table_to_lines(entry.get("table"))

    if mode == "all":
        parts: List[str] = []
        if pre:
            parts.append("PRE TEXT:\n" + "\n".join(pre))
        if table_lines:
            parts.append("TABLE:\n" + "\n".join(table_lines))
        if post:
            parts.append("POST TEXT:\n" + "\n".join(post))
        return "\n\n".join(parts) if parts else "No additional context provided."

    rng = random.Random(noisy_seed + example_index)
    gold_text_rows: List[int] = qa.get("ann_text_rows") or []
    gold_table_rows: List[int] = qa.get("ann_table_rows") or []
    pre_len = len(pre)
    pre_gold = [idx for idx in gold_text_rows if idx < pre_len]
    post_gold = [idx - pre_len for idx in gold_text_rows if idx >= pre_len]

    pre_lines = _sample_with_noise(pre, pre_gold, noisy_text_distractors, rng)
    post_lines = _sample_with_noise(post, post_gold, noisy_text_distractors, rng)
    table_selected = _sample_with_noise(table_lines, gold_table_rows, noisy_table_distractors, rng)

    parts: List[str] = []
    if pre_lines:
        parts.append("PRE TEXT:\n" + "\n".join(pre_lines))
    if table_selected:
        parts.append("TABLE:\n" + "\n".join(table_selected))
    if post_lines:
        parts.append("POST TEXT:\n" + "\n".join(post_lines))
    return "\n\n".join(parts) if parts else "No additional context provided."


def extract_target(qa: Dict, field: str) -> str:
    field_key = "exe_ans" if field == "numerical" else field
    target = qa.get(field_key, "")
    if isinstance(target, list):
        target = " ".join(target)
    if not target:
        fallback = qa.get("exe_ans") or qa.get("answer") or "unknown"
        target = fallback
    return str(target).strip()


def resolve_task_hint(target_field: str, task_instruction: str) -> str:
    hint = task_instruction.strip()
    if not hint:
        if target_field == "program":
            hint = "Generate the FinQA reasoning program (DSL) that answers the question."
        else:
            hint = "Return the final numeric answer."
    return hint


def build_prompt(question: str, context: str, hint: str) -> str:
    return f"{hint.strip()}\n\nQuestion:\n{question.strip()}\n\nContext:\n{context.strip()}\n\nAnswer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize FinQA noisy-gold prompts.")
    parser.add_argument("--input_file", type=Path, default=Path("FinQA/dataset/train.json"))
    parser.add_argument("--output_file", type=Path, default=Path("__output__/noisy_gold_prompts.jsonl"))
    parser.add_argument(
        "--input_mode",
        choices={"noisy_gold", "gold", "all"},
        default="noisy_gold",
        help="Context strategy; noisy_gold mixes gold evidence with random distractors.",
    )
    parser.add_argument(
        "--target_field",
        choices={"program", "numerical", "answer"},
        default="numerical",
    )
    parser.add_argument("--task_instruction", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of examples processed.")
    parser.add_argument("--noisy_text_distractors", type=int, default=3)
    parser.add_argument("--noisy_table_distractors", type=int, default=1)
    parser.add_argument("--noisy_context_seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input_file.read_text())
    if args.max_samples:
        data = data[: args.max_samples]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    hint = resolve_task_hint(args.target_field, args.task_instruction)

    with args.output_file.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(data):
            qa = entry["qa"]
            context = build_context(
                entry,
                args.input_mode,
                noisy_text_distractors=args.noisy_text_distractors,
                noisy_table_distractors=args.noisy_table_distractors,
                noisy_seed=args.noisy_context_seed,
                example_index=idx,
            )
            prompt = build_prompt(qa["question"], context, hint)
            record = {
                "id": entry["id"],
                "question": qa.get("question", ""),
                "context": context,
                "prompt": prompt,
                "target": extract_target(qa, args.target_field),
                "target_field": args.target_field,
            }
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(data)} prompts to {args.output_file}")


if __name__ == "__main__":
    main()
