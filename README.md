# FinQA Gemma Finetuning Utilities

This repository contains scripts to finetune Gemma-3-1B-IT on the FinQA dataset, generate FinQA DSL programs or numeric answers, and evaluate execution accuracy. It also includes utilities for noisy-evidence training and runtime tracking.

## Key scripts
- `scripts/finetune_gemma.py`: Main training/eval driver.
  - Modes:
    - `--target_field program|exe_ans|answer` (DSL program vs executed answer vs raw answer string).
    - `--input_mode gold|all|noisy_gold`:
      - `gold`: only annotated evidence (`gold_inds`); oracle context.
      - `all`: full pre_text/table/post_text.
      - `noisy_gold`: annotated evidence plus random distractors (see noise knobs below).
  - Noise knobs (for `noisy_gold`):
    - `--noisy_text_distractors`: number of non-evidence pre/post sentences to add.
    - `--noisy_table_distractors`: number of non-evidence table rows to add.
    - `--noisy_context_seed`: base seed for deterministic sampling per example.
  - Generation knobs: `--generation_max_new_tokens`, `--generation_num_beams`, `--generation_temperature`, `--generation_top_p`, `--generation_do_sample`.
  - Training knobs: standard HF Trainer arguments (epochs, LR, batch sizes, accumulation, scheduler, precision, etc.).
  - Outputs per run (under `__output__/.../`):
    - `args.json`: resolved arguments.
    - `metrics.jsonl`, `train_loss.log`, `eval_loss.log`: training/eval traces.
    - `final_model/`: saved model and tokenizer.
    - `test_predictions.jsonl`: per-example predictions (includes DSL/answer, prompt, and per-sample inference latency in ms).
    - `test_metrics.json`: aggregate metrics (execution/program/answer accuracy plus inference timing stats).
    - `test_inference_latencies.jsonl`: per-example generation latency in milliseconds.
    - `timing.json`: wall-clock durations for train/eval/test inference.
- `scripts/make_noisy_gold_dataset.py`: Materializes prompts with noisy-gold context for offline inspection/training. Writes JSONL with `id`, `question`, `context`, `prompt`, and `target`.

## Runtime tracking
- Training time: recorded as `train_duration_seconds` in `timing.json`.
- Eval time: `eval_duration_seconds` in `timing.json` and echoed into eval metrics.
- Test inference:
  - Per-sample latency (ms) is stored in `test_predictions.jsonl` and in `test_inference_latencies.jsonl`.
  - Aggregates in `test_metrics.json`: `inference_total_seconds`, `inference_latency_ms_{avg,p50,p90,p95,p99}`, and pointer to `latency_file`.

## Ready-made run scripts (`__scripts__/`)
- `gemma-input_gold-output_program/script.sh`: Oracle-evidence DSL training/eval (`input_mode gold`, `target_field program`).
- `gemma-input_all-output_program/script.sh`: Full-context DSL training/eval (`input_mode all`, `target_field program`).
- `gemma-input_noisy_gold-output_program/script.sh`: DSL training/eval with noisy-gold context (`input_mode noisy_gold`, distractor defaults set).

Run any script with `bash __scripts__/<run>/script.sh`. Outputs land under `__output__/<matching-name>*/`.

## Typical adjustments
- Shorten/expand context: switch `--input_mode` or adjust distractor counts.
- Precision: set `--bf16 true` or `--fp16 true` if your hardware supports it.
- Training budget: bump `--num_train_epochs`, `--per_device_train_batch_size`, or adjust `--learning_rate` as needed.
- Decoding: tweak beam/sampling settings via generation flags.

## Dataset note
FinQA splits live under `FinQA/dataset/` (`train.json`, `dev.json`, `test.json`). DSL definitions and evaluation follow the FinQA paper; execution accuracy is computed by running the predicted DSL against the normalized table/text fields.
