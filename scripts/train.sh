#!/bin/bash
# ================================
# train.sh
# Bash script to run Peptide2Mol training
# Supports default parameters with overrides and uses functions for clarity
# ================================

# -------------------------------
# Function: print_usage
# Description: Displays usage information
# -------------------------------
print_usage() {
    echo "Usage: bash scripts/train.sh [options]"
    echo ""
    echo "Environment variables (optional, default values shown):"
    echo "  EXPERIMENT       Default: mol_test       # Experiment name"
    echo "  DATA_DIR         Default: ./dataset/        # Directory containing LMDB files"
    echo "  LMDB_FILE        Default: data.pt   # LMDB filename"
    echo "  NUM_TRAIN        Default: 370000         # Number of training samples"
    echo "  BATCH_SIZE       Default: 8              # Batch size"
    echo "  LOG_DIR          Default: ./logs_train   # Directory for logs"
    echo "  LOGGER           Default: many_loggers   # Logger type"
    echo ""
    echo "Example:"
    echo "  bash scripts/train.sh"
    echo "  DATA_DIR=./demo/ LMDB_FILE=data_demo.pt BATCH_SIZE=4 LOG_DIR=./logs_demo bash scripts/train.sh"
}

# -------------------------------
# Function: run_training
# Description: Executes Peptide2Mol training with the specified parameters
# -------------------------------
run_training() {
    # Set default values if not already defined
    EXPERIMENT=${EXPERIMENT:-mol_test}
    DATA_DIR=${DATA_DIR:-"./dataset"}
    DATA_DIR=$(realpath "$DATA_DIR")
    LMDB_FILE=${LMDB_FILE:-"data.pt"}
    NUM_TRAIN=${NUM_TRAIN:-370000}
    BATCH_SIZE=${BATCH_SIZE:-8}
    LOG_DIR=${LOG_DIR:-"./logs_retrain"}
    LOGGER=${LOGGER:-"many_loggers"}

    echo "Running Peptide2Mol training with the following parameters:"
    echo "  EXPERIMENT = $EXPERIMENT"
    echo "  DATA_DIR   = $DATA_DIR"
    echo "  LMDB_FILE  = $LMDB_FILE"
    echo "  NUM_TRAIN  = $NUM_TRAIN"
    echo "  BATCH_SIZE = $BATCH_SIZE"
    echo "  LOG_DIR    = $LOG_DIR"
    echo "  LOGGER     = $LOGGER"
    echo ""

    # Execute the training command
    CUDA_VISIBLE_DEVICES=6 python src/train.py \
        experiment=$EXPERIMENT \
        ++paths.data_dir=$DATA_DIR \
        ++data.lmdb_fn=$LMDB_FILE \
        ++data.num_train=$NUM_TRAIN \
        ++data.batch_size=$BATCH_SIZE \
        ++paths.log_dir=$LOG_DIR \
        logger=$LOGGER
}

# -------------------------------
# Main script logic
# -------------------------------
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Call the training function
run_training
