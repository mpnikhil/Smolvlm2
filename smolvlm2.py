from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image, UnidentifiedImageError
import os

def load_and_prepare_image(image_path):
    """
    Load and prepare image for VLM processing.
    Supports multiple image formats (PNG, JPEG, WebP, etc.)
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Try to open and verify the image
        raw_image = Image.open(image_path)
        
        # Verify the image can be read
        raw_image.verify()
        
        # Re-open after verify (verify closes the image)
        raw_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')
            
        return raw_image
        
    except UnidentifiedImageError:
        raise ValueError(f"File is not a recognized image format: {image_path}")
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

# Initialize model and processor
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
model_path = "./SmolVLM2-500M-Video-Instruct-mpnikhil1"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
).to(device)

# Example usage
image_path = "testimage2.webp"  # Can be .jpg, .png, .webp, etc.
try:
    raw_image = load_and_prepare_image(image_path)
    
    messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a real estate analyst. Your task is to classify the kitchen in the image as either 'Updated' or 'Original.' "
                "Do not describe the image in detail. Only respond with one of the following formats:\n"
                "- 'Updated: [reason]'\n"
                "- 'Original: [reason]'\n"
                "- 'Kitchen not visible or insufficient information.'\n\n"
                "Examples:\n"
                "Image 1: Stainless steel appliances, quartz countertops, and modern cabinets → 'Updated: Modern appliances and countertops.'\n"
                "Image 2: Yellowed appliances, laminate countertops, and old cabinetry → 'Original: Outdated appliances and cabinets.'\n"
                "Image 3: No visible kitchen → 'Kitchen not visible or insufficient information.'\n\n"
                "Always choose only one response. Never provide detailed scene descriptions."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Is the kitchen updated or original? Respond with only 'Updated', 'Original', or 'Kitchen not visible.'"},
            {"type": "image", "path": raw_image} 
        ]
    },
]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])
    
except Exception as e:
    print(f"Error processing image: {str(e)}")