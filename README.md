# Kitchen Classification with Hugging Face SmolVLM2

## Overview
This project leverages the **Hugging Face SmolVLM2** vision-language model to classify kitchen images as **"Updated"** or **"Dated"**.  

### Primary Steps
1. **Dataset Preparation:** Download and process a dataset of kitchen images.  
2. **Fine-Tuning:** Fine-tune the SmolVLM2 model on the prepared dataset.  
3. **Local Inference:** Use the fine-tuned model for local inference.  

---

## Directory Structure
- `prepare_hf_dataset.py`: Script to prepare and upload the dataset to Hugging Face Hub.  
- `smolvlm2.py`: Main script for loading and configuring the SmolVLM2 model.  
- `finetune.py`: Script for fine-tuning the SmolVLM2 model on the prepared dataset.  
- `finetune-dataset.py`: Additional script for fine-tuning-specific preprocessing 
- `README.md`: This overview document.  

---

## Prerequisites

### Python
Ensure you have **Python 3.8** or higher installed.

### Hugging Face CLI
Install and log in using the following commands:

```bash
pip install huggingface_hub huggingface-cli
huggingface-cli login
```

### Required Packages
Install the following Python packages:

- `duckduckgo_search`  
- `Pillow`  
- `requests`  
- `datasets`  
- `transformers`  

---

## Steps to Run

### 1. Prepare Dataset
Prepare and upload the dataset to the Hugging Face Hub:

```bash
python prepare_hf_dataset.py --source_dir <path_to_downloaded_images> --dataset_name <your_username/kitchen-classifier>
```

Replace:
- `<path_to_downloaded_images>`: Path where images are stored.  
- `<your_username/kitchen-classifier>`: Desired dataset name.  

---

### 3. Fine-Tune Model
Fine-tune the SmolVLM2 model on the prepared dataset:

```bash
python finetune.py --dataset_name <your_username/kitchen-classifier>
```

---

### 4. Local Inference
Use the fine-tuned model for local inference:

```bash
python smolvlm2.py 
```
---

## Notes
- Ensure all required packages are installed before running the scripts.  
- Logs will be generated in `dataset_preparation.log` during dataset preparation.  

