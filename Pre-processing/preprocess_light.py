import cv2
import numpy as np
import os
from skimage import exposure

# Define input and output directories
input_dir = r"C:\Users\Hp\Documents\Capstone\Dataset\New folder"
output_dir = r"C:\Users\Hp\Documents\Capstone\Sem 6\Capstone\Pre-processing\preprocessed_xrays"
os.makedirs(output_dir, exist_ok=True)

# Parameters (reverted to initial contrast-focused settings)
IMG_SIZE = (512, 512)  # Adjust this to your original image size if known (e.g., 1024x1024)
BLUR_KERNEL = (3, 3)  # Minimal blur as in earlier versions
CLAHE_CLIP_LIMIT = 2.0  # Original high-contrast setting
CLAHE_TILE_SIZE = (8, 8)  # Original tile size for local enhancement
USE_THRESHOLDING = False  # Avoid binarization to retain details

# Function to preprocess a single image
def preprocess_image(image_path, output_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Resize image (use original size if possible, or adjust IMG_SIZE)
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)  # Cubic for better detail
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Minimal noise reduction
    img_blurred = cv2.GaussianBlur(img_normalized, BLUR_KERNEL, 0)
    
    # Contrast enhancement (reverted to high contrast)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    img_enhanced = clahe.apply((img_blurred * 255).astype(np.uint8))
    
    # Normalize again after enhancement
    img_final = img_enhanced / 255.0
    
    # Skip cropping to avoid pixel loss (commented out)
    # h, w = img_final.shape
    # if h > 2 * CROP_SIZE and w > 2 * CROP_SIZE:
    #     img_cropped = img_final[CROP_SIZE:h-CROP_SIZE, CROP_SIZE:w-CROP_SIZE]
    # else:
    #     img_cropped = img_final
    # img_cropped_resized = cv2.resize(img_cropped, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    
    # Use enhanced image directly as output (no cropping/resizing again)
    img_output = img_final
    
    # Optional thresholding (skipped here)
    if USE_THRESHOLDING:
        _, img_thresholded = cv2.threshold((img_output * 255).astype(np.uint8), 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_output = img_thresholded / 255.0
    
    # Save the preprocessed image
    cv2.imwrite(output_path, (img_output * 255).astype(np.uint8))
    
    return {
        'original': img_resized / 255.0,
        'enhanced': img_final,
        'output': img_output
    }

# Process all images in the directory
for idx, filename in enumerate(os.listdir(input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"preprocessed_{filename}")
        
        # Preprocess without displaying
        result = preprocess_image(image_path, output_path)
        if result is None:
            continue
        
        print(f"Processed {filename} ({idx + 1}/{len(os.listdir(input_dir))})")

print("Preprocessing completed!")