from datasets import Dataset, DatasetDict, Features, Value, Image
import json
from pathlib import Path
from PIL import Image as PILImage
import shutil
import os
from huggingface_hub import HfApi
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

def validate_image(image_path: Path) -> bool:
    """
    Validate if an image file exists and can be opened
    """
    try:
        if not image_path.exists():
            logging.warning(f"Missing image file: {image_path}")
            return False
            
        # Try to open and verify the image
        with PILImage.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.warning(f"Invalid image file {image_path}: {str(e)}")
        return False

def process_dataset_split(data: List[Dict], temp_dir: Path) -> Tuple[List[Dict], List[str]]:
    """
    Process a dataset split and track errors
    """
    processed_data = []
    errors = []
    
    for item in data:
        print(item)
        image_path = Path(item["image_path"])
        
        # Validate image
        if not validate_image(image_path):
            errors.append(f"Failed to process {image_path}")
            continue
            
        try:
            # Copy image to temp directory
            new_path = temp_dir / image_path.name
            shutil.copy2(image_path, new_path)
            
            # Create data entry matching the metadata format
            data_item = {
                "image": str(new_path),
                "question": item["text"],  # Changed from "question" to "text"
                "answer": "Updated" if item["response"] == "updated" else "Dated",  # Format response
                "split": "train" if "train" in str(image_path) else "validation"
            }
            processed_data.append(data_item)
            
        except Exception as e:
            errors.append(f"Error processing {image_path}: {str(e)}")
            continue
            
    return processed_data, errors

def prepare_hf_dataset(source_dir: str, dataset_name: str, log_file: str = "dataset_preparation.log"):
    """
    Prepare downloaded kitchen dataset for HuggingFace Hub with error handling
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    source_dir = Path(source_dir)
    
    # Validate source directory
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Load and validate metadata
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found: {metadata_path}")
        
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {str(e)}")
    
    # Create temporary directory
    temp_dir = Path("hf_dataset_temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Split data based on path containing "train" or "val"
    train_items = [item for item in metadata if "train" in item["image_path"]]
    val_items = [item for item in metadata if "val" in item["image_path"]]
    
    logging.info(f"Found {len(train_items)} training images and {len(val_items)} validation images")
    
    # Process splits
    print("Processing training split...")
    train_data, train_errors = process_dataset_split(train_items, temp_dir)
    print("Processing validation split...")
    val_data, val_errors = process_dataset_split(val_items, temp_dir)
    
    # Log processing results
    all_errors = train_errors + val_errors
    if all_errors:
        logging.warning("Encountered errors processing images:")
        for error in all_errors:
            logging.warning(error)
            
    logging.info(f"Successfully processed {len(train_data)} training and {len(val_data)} validation images")
    
    if not train_data or not val_data:
        raise ValueError("No valid images found in one or both splits")
    
    # Create HuggingFace datasets
    features = Features({
        "image": Image(),
        "question": Value("string"),
        "answer": Value("string"),
        "split": Value("string")
    })
    
    try:
        print("Creating HuggingFace datasets...")
        train_dataset = Dataset.from_list(train_data, features=features)
        val_dataset = Dataset.from_list(val_data, features=features)
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    except Exception as e:
        raise ValueError(f"Failed to create dataset: {str(e)}")
    
    # Create dataset card with error statistics
    success_rate = (len(train_data) + len(val_data)) / (len(train_items) + len(val_items)) * 100
    
    dataset_card = f"""---
license: mit
task_categories:
  - image-classification
  - visual-question-answering
language:
  - en
size_categories:
  - 1K<n<10K
---

# Kitchen Classification Dataset

This dataset contains images of kitchens labeled as either "Updated" or "Dated" for fine-tuning vision-language models.

## Dataset Description

### Dataset Summary

The dataset consists of kitchen images collected for training a model to classify kitchens as either updated/modern or dated/outdated.

- Train split: {len(train_data)} images (from {len(train_items)} original)
- Validation split: {len(val_data)} images (from {len(val_items)} original)
- Overall success rate: {success_rate:.1f}%

### Data Fields

- image: The kitchen image
- question: The classification question
- answer: The label ("Updated" or "Dated")
- split: Dataset split ("train" or "validation")

### Data Splits

The data is split into training (80%) and validation (20%) sets.

### Processing Notes

- Total images processed: {len(train_data) + len(val_data)}
- Images with errors: {len(all_errors)}
- Processing date: {datetime.now().strftime('%Y-%m-%d')}
"""
    
    # Save dataset card
    with open(temp_dir / "README.md", "w") as f:
        f.write(dataset_card)
    
    # Push to hub
    logging.info(f"Pushing dataset to HuggingFace Hub as {dataset_name}...")
    try:
        dataset_dict.push_to_hub(
            dataset_name,
            private=True
        )
    except Exception as e:
        raise ValueError(f"Failed to upload dataset: {str(e)}")
    
    # Clean up
    logging.info("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    logging.info(f"Dataset successfully uploaded to HuggingFace Hub: {dataset_name}")
    logging.info(f"Check the log file at {log_file} for detailed processing information")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Prepare and upload dataset to HuggingFace Hub')
    parser.add_argument('--source_dir', type=str, default='kitchen_dataset',
                        help='Directory containing the downloaded dataset')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name for the dataset on HuggingFace Hub (e.g. username/kitchen-classifier)')
    parser.add_argument('--log_file', type=str, default='dataset_preparation.log',
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Check HuggingFace login
    try:
        api = HfApi()
        api.whoami()
    except Exception as e:
        print("Please login to HuggingFace first using:")
        print("huggingface-cli login")
        exit(1)
        
    try:
        prepare_hf_dataset(args.source_dir, args.dataset_name, args.log_file)
    except Exception as e:
        logging.error(f"Failed to prepare dataset: {str(e)}")
        exit(1)