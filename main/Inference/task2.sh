#!/bin/bash

SCRIPT_PATH="../LLaVA_Finetune/infer.sh"

if [[ -f "$SCRIPT_PATH" ]]; then
    chmod +x "$SCRIPT_PATH"
    bash "$SCRIPT_PATH"
else
    echo "Error: $SCRIPT_PATH not found!"
    exit 1
fi
