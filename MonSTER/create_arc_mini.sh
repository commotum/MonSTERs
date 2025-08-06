#!/bin/bash

# Set source and destination directories
TRAIN_SRC="/home/jake/Developer/MonSTERs/dataset/raw-data/ARC-AGI-2/data/training"
EVAL_SRC="/home/jake/Developer/MonSTERs/dataset/raw-data/ARC-AGI-2/data/evaluation"
TRAIN_DEST="/home/jake/Developer/MonSTERs/dataset/raw-data/ARC-MINI/data/training"
EVAL_DEST="/home/jake/Developer/MonSTERs/dataset/raw-data/ARC-MINI/data/evaluation"

# Create destination directories if they don't exist
mkdir -p "$TRAIN_DEST"
mkdir -p "$EVAL_DEST"

# Check if source directories exist
if [ ! -d "$TRAIN_SRC" ]; then
    echo "Error: Training source directory $TRAIN_SRC does not exist."
    exit 1
fi
if [ ! -d "$EVAL_SRC" ]; then
    echo "Error: Evaluation source directory $EVAL_SRC does not exist."
    exit 1
fi

# Select 100 random JSON files from training set and copy to destination
find "$TRAIN_SRC" -type f -name "*.json" | shuf -n 100 | xargs -I {} cp {} "$TRAIN_DEST"
echo "Copied 100 random training puzzles to $TRAIN_DEST"

# Select 100 random JSON files from evaluation set and copy to destination
find "$EVAL_SRC" -type f -name "*.json" | shuf -n 100 | xargs -I {} cp {} "$EVAL_DEST"
echo "Copied 100 random evaluation puzzles to $EVAL_DEST"