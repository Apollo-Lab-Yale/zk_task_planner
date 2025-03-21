#!/usr/bin/env python3
"""
Test script for Molmo-7B-D-0924 multimodal model from Hugging Face.
This script loads the model, processes an image, and generates a description.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image, ImageStat
import requests
import time
import argparse

def process_image(image_url):
    """Process an image, handling transparency by adding a background if needed."""
    # Load the image
    print(f"Downloading image from: {image_url}")
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        # For transparent images, add an appropriate background
        if image.mode == 'RGBA':
            # Calculate brightness to determine background color
            gray_image = image.convert('L')
            stat = ImageStat.Stat(gray_image)
            average_brightness = stat.mean[0]
            
            # Select background color based on image brightness
            bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)
            
            # Create new image with background
            new_image = Image.new('RGB', image.size, bg_color)
            new_image.paste(image, (0, 0), image)
            image = new_image
        else:
            # For other non-RGB formats
            image = image.convert("RGB")
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Test Molmo-7B-D-0924 model with an image.')
    parser.add_argument('--image_url', type=str, 
                        default="https://picsum.photos/id/237/536/354",
                        help='URL of the image to analyze')
    parser.add_argument('--prompt', type=str, 
                        default="Describe this image in detail.",
                        help='Prompt to send to the model')
    parser.add_argument('--max_tokens', type=int, 
                        default=200, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--use_bf16', action='store_true',
                        help='Use BFloat16 precision to reduce memory usage')
    
    args = parser.parse_args()
    
    # Install dependencies if needed
    # Uncomment these lines if you need to install dependencies
    # import subprocess
    # subprocess.check_call(["pip", "install", "einops", "torchvision"])
    
    start_time = time.time()
    print("Loading Molmo-7B-D-0924 model and processor...")
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Process the image
    image = process_image(args.image_url)
    
    # Process the image and text
    start_time = time.time()
    print(f"Processing image with prompt: '{args.prompt}'")
    
    inputs = processor.process(
        images=[image],
        text=args.prompt
    )
    
    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Use BFloat16 if requested
    if args.use_bf16:
        print("Using BFloat16 precision for inference")
        model.to(dtype=torch.bfloat16)
        inputs["images"] = inputs["images"].to(torch.bfloat16)
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=args.max_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    else:
        # Use autocast for efficient inference while maintaining precision
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                           enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=args.max_tokens, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
    
    # Only get generated tokens; decode them to text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"\nInference completed in {time.time() - start_time:.2f} seconds")
    print("\n--- Generated Response ---")
    print(generated_text.strip())
    print("-------------------------")

if __name__ == "__main__":
    main()