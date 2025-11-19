python scripts/finetune_gemma.py \
  --input_mode gold \
  --target_field program \
  --do_eval \
  --eval_steps 200 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1
