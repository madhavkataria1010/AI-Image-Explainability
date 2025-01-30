# AI-Generated Image Detection & Explanation  

## Overview  
With the rapid rise of AI-generated content from models like Stable Diffusion, MidJourney, and Adobe Firefly, ensuring content authenticity has become a critical challenge. This project addresses this issue by developing a model that accurately detects AI-generated images and provides human-interpretable explanations for its classifications.  

The system highlights distinguishing artifacts such as texture inconsistencies, unnatural lighting, and anomalous patterns to differentiate between real and synthetic images. This work was developed as part of the **InterIIT Tech Meet - Adobe Challenge**, focusing on AI authenticity and explainability.  

## Features  
✅ Detects AI-generated images with high accuracy  
✅ Highlights key artifacts contributing to classification  
✅ Provides interpretable explanations for decisions  
✅ Supports various generative models like Stable Diffusion and MidJourney  
✅ Enhances content authenticity and trust in digital media  

# Directory Structure

    adobe/
    ├── data/
    │   ├── task1/                  # Data and Script for Task 1 Data Generation
    │   ├── task2/                  # Data and its generation script for Task2
    │   
    |                                 
    ├── experiments/
    │   ├── task1_scripts/          # Approaches for Task 1
    │   └── task2_scripts/          # Approaches for Task 2
    ├── main/
    │   ├── Inference/
    │   │   ├── task1.sh            # Script for Task 1 inference
    │   │   └── task2.sh            # Script for Task 2 inference
    │   ├── LLaVA_Finetune
    │   |── Training/
    │   |    ├── task1_training.py  # Script for training Task 1 model
    │   |__finetune.sh               # Script for finetuning VLM 
    |   
    ├── models/
    │   └── task1_model.pth         # Trained model for Task 1
    ├── results/
    │   ├── 73_task1.json           # Output JSON for Task 1
    │   └── 73_task2.json           # Output JSON for Task 2
    ├── test/
    │   └── images/                 # Placeholder for input images for inference
    └── README.md                   # Documentation for the project

----------------------------
# Quick Setup for Inference
----------------------------

## This guide will help you run inference for both Task 1 and Task 2.

### Step 1: Prepare Input Images
    Place the images on which inference is required in the test/input_images/ directory.
    
### Important:
    1.Ensure that you place only PNG image files in the test/input_images/ folder.
    2.Do not add subfolders or non-image files.

### IMPORTANT: DOWNLOAD MODEL WEIGHTS
    1.Go to the following link: https://huggingface.co/team73/llava_finetuned
    2.Download the entire folder "llava-ftmodel"
    3.Place the downloaded weights inside the "LLaVA_Finetune" folder
    4.The final structure of "LLaVA_Finetune" folder should look like this:
    LLaVA_Finetune/
    ├── Dice                        # Folder              
    ├── LLaVA                       # Folder
    ├── llava-ftmodel               # Folder       
    ├── slurm-logs                  # Folder
    ├── test_adobe                  # Folder
    ├── final.json                  # json file       
    ├── fix.py                      # python file
    ├── infer.sh                    # shell script
    ├── inference.py                # python file
    ├── setup.py                    # python file       
    └── test.py                     # python file 
    

### Step 2: Run Inference
    1.Navigate to the main/Inference/ directory.
    2.Follow the instructions provided in the README.md file inside the respective directory to run the inference scripts for Task 1 or Task 2.

### Step 3: Review Results
#### After running the inference scripts, the results will be available in the results/ directory:
    task1_results.json: Contains output for Task 1.
    task2_results.json: Contains output for Task 2.



------------------------------------
# Quick Setup for Dataset Generation 
------------------------------------

This guide will help you generate data for task1 and task2 

### Step 1: Generate Data for Task 1
#### Navigate to the directory:

    
    cd data/task1/
      
#### Follow the instructions provided in the README.md located in the data/task1/Scripts/ directory.

### Step 2: Generate Data for Task 2
#### Navigate to the directory:

    
    cd data/task2/
    
    
#### Follow the instructions provided in the README.md located in the data/task2/reason_data/ directory.

-------------------------------------------
# Quick Setup for Training and Fine-tuning
-------------------------------------------

This guide will help you train and fine-tune for task1 and task2 

### Step 1: Train model for Task 1
#### Navigate to the directory:

    
    cd adobe/main/Training/task1
      
#### Follow the instructions provided in the README.md located in the main/Training/task1/  directory.

### Step 2: Finetune model for Task 2
#### Navigate to the directory:

    
    cd adobe/main
    
    
### Step 3: Start finetuning 

```bash

chmod +x finetune.sh
bash finetune.sh

```
#### Follow the instructions provided in the README.md located in the data/task2/reason_data/ directory.



