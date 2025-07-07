#!/usr/bin/env python3
"""
Comprehensive Urdu OCR Dataset Creator
=====================================

This script creates diverse training datasets for Tesseract OCR training.
Edit the configuration variables below to customize your dataset.

Features:
- Multiple font support (Naskh, Nastaleeq, Tehreer)
- Image variations (blur, noise, brightness, contrast)
- Configurable dataset sizes
- Automatic file generation for Tesseract training

Usage:
    python3 create_dataset.py
"""

import os
import random
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES TO CUSTOMIZE YOUR DATASET
# =============================================================================

# Dataset Configuration
DATASET_SIZE = 20000          # Total number of samples to generate
DATASET_NAME = "urdu_20k_mixed_dataset"  # Output folder name

# Font Configuration - Add your font paths here
FONTS = {
    'naskh': '/System/Library/Fonts/Supplemental/GeezaPro.ttc',
    'nastaleeq': '/System/Library/Fonts/Supplemental/GeezaPro.ttc', 
    'tehreer': '/System/Library/Fonts/Supplemental/GeezaPro.ttc'
}

# Font Distribution (should add up to 1.0)
FONT_DISTRIBUTION = {
    'naskh': 0.4,      # 40% Naskh
    'nastaleeq': 0.35, # 35% Nastaleeq  
    'tehreer': 0.25    # 25% Tehreer
}

# Image Variations Distribution (should add up to 1.0)
VARIATION_DISTRIBUTION = {
    'original': 0.3,        # 30% original
    'blur': 0.15,          # 15% blurred
    'noise': 0.15,         # 15% noisy
    'dark': 0.1,           # 10% dark
    'bright': 0.1,         # 10% bright
    'low_contrast': 0.1,   # 10% low contrast
    'high_contrast': 0.05, # 5% high contrast
    'sharp': 0.05          # 5% sharpened
}

# Image Settings
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 100
FONT_SIZE = 48
BACKGROUND_COLOR = (255, 255, 255)  # White
TEXT_COLOR = (0, 0, 0)              # Black

# Text Sources - Add your Urdu text here
URDU_TEXTS = [
    "یہ اردو زبان کا متن ہے",
    "تعلیم بہت اہم ہے",
    "کتاب پڑھنا مفید ہے", 
    "اردو ایک خوبصورت زبان ہے",
    "علم حاصل کرنا ضروری ہے",
    "محنت کا پھل میٹھا ہوتا ہے",
    "وقت بہت قیمتی ہے",
    "صبر کا پھل میٹھا ہوتا ہے",
    "دوستی ایک نعمت ہے",
    "سچائی ہمیشہ جیتتی ہے"
]

# =============================================================================
# DATASET GENERATION FUNCTIONS
# =============================================================================

def create_base_image(text, font_path, font_name):
    """Create a base image with text"""
    try:
        # Create image
        img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        
        # Load font
        font = ImageFont.truetype(font_path, FONT_SIZE)
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (IMAGE_WIDTH - text_width) // 2
        y = (IMAGE_HEIGHT - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, font=font, fill=TEXT_COLOR)
        
        return img
    except Exception as e:
        print(f"Error creating base image: {e}")
        return None

def apply_variation(img, variation_type):
    """Apply image variation"""
    if variation_type == 'original':
        return img
    elif variation_type == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif variation_type == 'noise':
        # Add noise
        np_img = np.array(img)
        noise = np.random.randint(0, 50, np_img.shape, dtype=np.uint8)
        noisy_img = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    elif variation_type == 'dark':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.6)
    elif variation_type == 'bright':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.4)
    elif variation_type == 'low_contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.5)
    elif variation_type == 'high_contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.8)
    elif variation_type == 'sharp':
        return img.filter(ImageFilter.SHARPEN)
    else:
        return img

def weighted_choice(choices, weights):
    """Choose item based on weights"""
    total = sum(weights.values())
    r = random.uniform(0, total)
    upto = 0
    for choice, weight in weights.items():
        if upto + weight >= r:
            return choice
        upto += weight
    return list(choices.keys())[-1]

def create_dataset():
    """Create the complete dataset"""
    print(f"🚀 Creating {DATASET_SIZE} samples in '{DATASET_NAME}'...")
    
    # Create output directory
    if os.path.exists(DATASET_NAME):
        shutil.rmtree(DATASET_NAME)
    os.makedirs(DATASET_NAME)
    
    # Generate samples
    for i in range(DATASET_SIZE):
        if i % 1000 == 0:
            print(f"📊 Generated {i}/{DATASET_SIZE} samples...")
        
        # Choose font and variation
        font_name = weighted_choice(FONTS, FONT_DISTRIBUTION)
        variation = weighted_choice(VARIATION_DISTRIBUTION, VARIATION_DISTRIBUTION)
        
        # Choose random text
        text = random.choice(URDU_TEXTS)
        
        # Create filename
        filename = f"{font_name}_{i:06d}_{variation}"
        
        # Create base image
        img = create_base_image(text, FONTS[font_name], font_name)
        if img is None:
            continue
            
        # Apply variation
        img = apply_variation(img, variation)
        
        # Save files
        img_path = os.path.join(DATASET_NAME, f"{filename}.tif")
        gt_path = os.path.join(DATASET_NAME, f"{filename}.gt.txt")
        
        # Save image
        img.save(img_path, 'TIFF')
        
        # Save ground truth
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"✅ Dataset creation complete!")
    print(f"📁 Location: {DATASET_NAME}/")
    print(f"📊 Total samples: {DATASET_SIZE}")
    
    # Create info file
    info_path = os.path.join(DATASET_NAME, "dataset_info.md")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"# Dataset Information\n\n")
        f.write(f"- **Total Samples**: {DATASET_SIZE}\n")
        f.write(f"- **Fonts**: {list(FONTS.keys())}\n")
        f.write(f"- **Variations**: {list(VARIATION_DISTRIBUTION.keys())}\n")
        f.write(f"- **Image Size**: {IMAGE_WIDTH}x{IMAGE_HEIGHT}\n")
        f.write(f"- **Font Size**: {FONT_SIZE}\n")

if __name__ == "__main__":
    create_dataset()
