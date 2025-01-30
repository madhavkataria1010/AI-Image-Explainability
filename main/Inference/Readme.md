
# Inference on Images 

### Generating the JSON file output for task1 : This step is handled by a script from the task1 directory.
### Running inference with VLM: This step uses the generated JSON file and the corresponding images to generate relevant artifacts.
### The instructions below guide you through setting up the environment, running the necessary scripts, and troubleshooting common issues.

# Directory Structure

    Inference/
    │
    ├── task1/                    
    │   └── script1.py             # Script that generates the JSON output
    ├── task1.sh                   # Shell script for running task1 (generating JSON)
    ├── task2.sh                   # Shell script for running task2 (VLM inference)
    ├── task1.yml                  # Conda environment YAML 
    ├── README.md                  # This file
    └── ...



## Step 1: Set up the Conda environment for task1

### Make sure you are in directory: adobe/main/Inference/
### Create the Conda environment for task1: To create the environment using the provided YAML file, run:

```bash

conda env create -f task1.yml

```

### Activate the task1 environment: After the environment is created, activate it using:


```bash

conda activate adobe_new

```
## Step 2: Execute task1

```bash

chmod +x task1.sh
bash task1.sh

```

## Step 3: Set up the Conda environment for task2

### Note: Change the following command according to present working directory . Pls make sure that you are in the following folder "/adobe/main/LLaVA_Finetune/LLaVA" before conda environment setup

```bash 
cd  adobe/main/LLaVA_Finetune/LLaVA

```


### Create the Conda environment for task2: run:

```bash

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

```

### After successful completion of above commands : run:


```bash

pip install -e ".[train]"
pip install flash-attn --no-build-isolation

```
### Note that flash attention is not required for inference purposes, but we need it for fine-tuning. Please make sure to use hardware compatible with flash attention while running the script for fine-tuning.


## Step 4: Execute task2

## Make sure your present working directory is main/Inference/

```bash

chmod +x task2.sh
bash task2.sh

```

### Upon successful execution, you will see the results saved in results directory in root folder


# Troubleshooting
### Missing Conda environment: If the Conda environment fails to activate, ensure you have run the correct conda env create -f command for both environments.
### Paths to Images or JSON: Ensure the paths to your images/ folder and test_info.json are correct when running infer.sh.
### CUDA Issues: If using GPU, verify that CUDA is installed and properly configured. You may need to install the correct version of CUDA for the environment.
### JSON Format: If the generated test_info.json or predictions.json file is not properly formatted, ensure that the scripts generating them are executed correctly.

---