# GenImage Dataset Preparation and Downsampling

#### This document provides detailed steps to set up and process the GenImage dataset, including downloading, downsampling, and selecting experimental data for training purposes. The dataset includes images from  generators such as Midjourney, Stable Diffusion, StyleGAN and BigGAN.

## Steps to Download GenImage Dataset

## 1. Dataset Selection

### For our experimentation, we chose images generated using:

- BigGAN
- Stable Diffusion
- Midjourney
- StyleGAN  (Downloaded from Kaggle. Refer Notes Section)

## 2. Install and Configure rclone

To handle the large dataset (over 200GB), we use rclone for efficient downloading.Install rclone.Follow the installation guide on the official rclone site.Configure rclone for Google Drive

### Run the following command to set up access to Google Drive:

```bash
rclone config
```

### Ensure shared drives are enabled during the configuration.

## 3. Download the Dataset

Use the following command to download the dataset from the shared Google Drive:

```bash
rclone copy "gdrive:SharedDrive/genimage/<dataset_name>" "/path/to/local/destination/" --drive-shared-with-me --dry-run
```

#### Remove --dry-run to execute the actual download.


## 4. Combine Multipart ZIP Files

### If the dataset is split into multiple parts, merge them into a single ZIP file:

```bash
zip -f <dataset_folders_name>.zip --out <dataset_folders_name>.zip
```

## 5. Extract the ZIP File

Unzip the combined file:

```bash
unzip <dataset_folders_name>.zip -d ./training_data
```

## 5.Downsampling and Selection

### Script: downsampling_selection.py

#### This script is used for:
Downsampling: Reducing image resolution as per project requirements.\
Selection: Filtering and selecting specific images from the custom dataset.

### Steps to Run the Script

Set Up the Conda Environment.\
Create and activate the conda environment using the provided YAML file:

```bash
conda env create -f conda.yml
conda activate yolo_torch_env
```

Execute the Script\
Run the following command to downsample and select images:

```bash
python downsampling_selection.py
```

## Notes

### System Requirements: Ensure you have sufficient disk space (>200GB) for downloading and processing the dataset.
### Readme Files: For additional information, refer to respective Readme files in the Scripts folder.
### StyleGAN images can be installed from kaggle https://www.kaggle.com/datasets/awsaf49/artifact-dataset?utm_source=chatgpt.com
---