#!/bin/bash

# ================================
# run_inference.sh
# Bash script to run Peptide2Mol inference
# Supports default parameters with overrides and uses functions for clarity
# ================================

# -------------------------------
# Function: print_usage
# Description: Displays script usage information
# -------------------------------
print_usage() {
    echo "Usage: ./run_inference.sh [options]"
    echo ""
    echo "Environment variables (optional, default values shown):"
    echo "  EXPERIMENT       Default: mol_test       # Experiment name"
    echo "  CKPT_PATH        Default: ./ckpts/PMT_major.ckpt  # Checkpoint path"
    echo "  DATA_DIR         Default: ./poc_test/   # Directory with .pt files"
    echo "  LMDB_FILE        Default: 5wbj_MTOR_poc.pt  # LMDB filename"
    echo "  MODEL            Default: Moldiff_test  # Model name"
    echo "  BATCH_SIZE       Default: 2             # Batch size for inference"
    echo "  LOG_DIR          Default: ./logs  # Directory for logs"
    echo "  SAMPLE_PDB_DIR   Default: ./poc_test    # Directory for input pdbs"
    echo "  SAMPLE_LOG_DIR   Default: ./sample_test  # Directory for sample outputs"
    echo ""
    echo "Example:"
    echo "  ./run_inference.sh"
    echo "  EXPERIMENT=mol_test CKPT_PATH=/path/to/ckpt.ckpt MODEL=Moldiff_test_partial BATCH_SIZE=1 ./run_inference.sh"
}

# -------------------------------
# Function: run_inference
# Description: Executes Peptide2Mol inference with the specified parameters
# -------------------------------
run_inference() {
    # Set default values if not already defined
    EXPERIMENT=${EXPERIMENT:-mol_test}
    CKPT_PATH=${CKPT_PATH:-"$PWD/ckpts/PMT_major.ckpt"}
    DATA_DIR=${DATA_DIR:-"$PWD/poc_test/"}
    LMDB_FILE=${LMDB_FILE:-"5wbj_MTOR_poc.pt"}
    MODEL=${MODEL:-"Moldiff_test"}
    BATCH_SIZE=${BATCH_SIZE:-2}
    LOG_DIR=${LOG_DIR:-"./logs"}
    SAMPLE_PDB_DIR=${SAMPLE_PDB_DIR:-"$PWD/poc_test"}
    SAMPLE_LOG_DIR=${SAMPLE_LOG_DIR:-"$PWD/sample_test"}

    echo "Running Peptide2Mol inference with the following parameters:"
    echo "  EXPERIMENT      = $EXPERIMENT"
    echo "  CKPT_PATH       = $CKPT_PATH"
    echo "  DATA_DIR        = $DATA_DIR"
    echo "  LMDB_FILE       = $LMDB_FILE"
    echo "  MODEL           = $MODEL"
    echo "  BATCH_SIZE      = $BATCH_SIZE"
    echo "  LOG_DIR         = $LOG_DIR"
    echo "  SAMPLE_PDB_DIR  = $SAMPLE_PDB_DIR"
    echo "  SAMPLE_LOG_DIR  = $SAMPLE_LOG_DIR"
    echo ""

    # Execute the inference command
    python src/eval.py \
      experiment=$EXPERIMENT \
      ckpt_path=$CKPT_PATH \
      ++paths.data_dir=$DATA_DIR \
      +data.lmdb_fn=$LMDB_FILE \
      data=mol_test_true \
      model=$MODEL \
      data.infer_batch_size=$BATCH_SIZE \
      trainer.devices=1 \
      ++paths.log_dir=$LOG_DIR \
      ++model.net.sample.log_dir=$SAMPLE_LOG_DIR \
      ++model.net.sample.pdb_dir=$SAMPLE_PDB_DIR \
      ++model.net.sample.batch_size=1
}

# -------------------------------
# Main script logic
# -------------------------------
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Call the inference function
run_inference
