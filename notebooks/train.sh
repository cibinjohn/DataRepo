#!/bin/bash
source /home/cibin/virtualenvs/transformers/bin/activate

cd ..

export TASK_NAME="sst2"
export OUTPUT_DIR="CHECKPOINT/SAMPLE/"
export train_file="DATA/sample/train.csv"
export test_file="DATA/sample/train.csv"
export validation_file="DATA/sample/train.csv"

python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT_DIR

  #> STATUS/train_status.txt 2> STATUS/train_error.txt


