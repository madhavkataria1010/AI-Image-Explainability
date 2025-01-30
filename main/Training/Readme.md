
# Training on Images 

### The instructions below guide you through setting up the environment, running the necessary scripts, and troubleshooting common issues.



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
## Step 2: Execute  training for task1
### Make sure you are in directory: adobe/main/Training/

```bash

python train.py

```
