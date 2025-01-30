#!/bin/bash

date;hostname;pwd


# module load python/3.10
nvidia-smi
bash LLaVA_Finetune/LLaVA/scripts/v1_5/finetune_task_lora.sh


