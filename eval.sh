#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <CKPT_BASENAME> <LMDB_FN>"
    exit 1
fi

# read config
CKPT_BASENAME=$1
LMDB_FN=$2
LMDB_NAME="${LMDB_FN%.*}"

EXPERIMENT="mol_test"
CKPT_PATH="$PWD/logs/train_Moldiff_test/runs/2025-06-02_15-43-27/checkpoints/$CKPT_BASENAME"
DATA_DIR="/home/chengxi/data/work/test/comp2404/PMT/PMT_250421/peptide2mol/eval/testcase"
DATA="mol_test_true"
MODEL="Moldiff_test"
INFER_BATCH_SIZE=2
DEVICES=1
LOG_DIR="./logs_eval"
SAMPLE_LOG_DIR="/home/chengxi/data/work/test/comp2404/PMT/PMT_250421/peptide2mol/eval/sample/sample_05/$LMDB_NAME"
PDB_DIR="/home/chengxi/data/work/test/comp2404/PMT/PMT_250421/peptide2mol/eval/testcase"
SAMPLE_BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=1 python src/eval.py \
  experiment=$EXPERIMENT \
  ckpt_path=$CKPT_PATH \
  ++paths.data_dir=$DATA_DIR \
  +data.lmdb_fn=$LMDB_FN \
  data=$DATA \
  model=$MODEL \
  data.infer_batch_size=$INFER_BATCH_SIZE \
  trainer.devices=$DEVICES \
  ++paths.log_dir=$LOG_DIR \
  ++model.net.sample.log_dir=$SAMPLE_LOG_DIR \
  ++model.net.sample.pdb_dir=$PDB_DIR \
  ++model.net.sample.batch_size=$SAMPLE_BATCH_SIZE
