#!/bin/bash

# Define paths
DATA_DIR="../../test/images/"
MODEL_PATH="../../models/train.pth"
OUTPUT_PATH="../../results/73_task1.json"

# Run the Python evaluation script
python task1/eval.py -d "$DATA_DIR" -m "$MODEL_PATH" -o "$OUTPUT_PATH"
