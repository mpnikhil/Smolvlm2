from duckduckgo_search import DDGS
import requests
from PIL import Image
import os
import json
from pathlib import Path
from io import BytesIO
import time

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize if too large
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        img.save(save_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def download_kitchen_images():
    # Search terms
    updated_terms = [
        'modern kitchen 2024',
        'contemporary kitchen design',
        'renovated kitchen after',
        'luxury modern kitchen',
        'updated kitchen remodel',
        'modern kitchen renovation',
        'kitchen remodel after photos',
        'contemporary kitchen renovation',
        'new kitchen design 2024',
        'modern kitchen makeover'
    ]
    
    dated_terms = [
        'old outdated kitchen',
        '1990s kitchen design',
        'dated kitchen before renovation',
        'old kitchen before remodel',
        'outdated kitchen cabinets',
        'old kitchen photos',
        'vintage kitchen 1980s',
        '1970s kitchen design',
        'old fashioned kitchen',
        'kitchen before renovation'
    ]
    
    # Create directories
    base_dir = Path("kitchen_dataset")
    for split in ['train', 'val']:
        for label in ['updated', 'dated']:
            (base_dir / split / label).mkdir(parents=True, exist_ok=True)

    dataset = []
    ddgs = DDGS()
    
    # Process each category
    for label, search_terms in [('updated', updated_terms), ('dated', dated_terms)]:
        count = 0
        print(f"\nDownloading {label} kitchen images...")
        
        for term in search_terms:
            print(f"\nSearching for: {term}")
            
            # Search for images
            try:
                results = list(ddgs.images(
                            term,
                            max_results=30
                        ))
                
                for i, result in enumerate(results):
                    if count >= 300:  # Limit to 300 per category
                        break
                        
                    image_url = result['image']
                    split = 'val' if count % 5 == 0 else 'train'
                    new_name = f"{label}_{count}.jpg"
                    save_path = base_dir / split / label / new_name
                    
                    if download_image(image_url, save_path):
                        dataset.append({
                            "file_name": new_name,
                            "image_path": str(save_path),
                            "text": "Is this kitchen updated or dated?",
                            "response": label
                        })
                        count += 1
                        print(f"Downloaded {count} {label} kitchen images", end='\r')
                    
                    # Small delay to be nice to the server
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error with search term '{term}': {str(e)}")
                continue

    # Save metadata
    with open(base_dir / "metadata.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDownload complete! Total images: {len(dataset)}")
    
    # Print split statistics
    train_count = len([x for x in dataset if 'train' in x['image_path']])
    val_count = len([x for x in dataset if 'val' in x['image_path']])
    print(f"\nSplit statistics:")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

if __name__ == "__main__":
    # Install required packages if not already installed
    import subprocess
    import sys
    
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    required_packages = ['duckduckgo_search', 'Pillow', 'requests']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            install(package)
    
    download_kitchen_images()