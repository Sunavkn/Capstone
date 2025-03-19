
import cv2
import numpy as np
import os
from skimage import exposure

# Define input and output directories
input_dir = r"C:\Users\jayan\SEM 6\Capstone\TB_Chest_Radiography_Database\Tuberculosis"
output_dir = r"C:\Users\jayan\SEM 6\Capstone\TB_Chest_Radiography_Database\tb_preprocess_1"
os.makedirs(output_dir, exist_ok=True)

# Parameters
IMG_SIZE = (256, 256)  # Standard size for resizing
CROP_SIZE = 1  # Pixels to crop from each side (adjust based on your X-rays)

# Function to preprocess a single image
def preprocess_image(image_path, output_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Resize image
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Minimal noise reduction (smaller kernel to preserve details)
    img_blurred = cv2.GaussianBlur(img_normalized, (3, 3), 0)  # Reduced from (5, 5)
    
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply((img_blurred * 255).astype(np.uint8))
    
    # Normalize again after enhancement
    img_final = img_enhanced / 255.0
    
    # Cropping (remove borders)
    h, w = img_final.shape
    if h > 2 * CROP_SIZE and w > 2 * CROP_SIZE:  # Ensure cropping is possible
        img_cropped = img_final[CROP_SIZE:h-CROP_SIZE, CROP_SIZE:w-CROP_SIZE]
    else:
        img_cropped = img_final  # Skip cropping if image is too small
    
    # Resize back to IMG_SIZE after cropping
    img_cropped_resized = cv2.resize(img_cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Edge detection (improved thresholds)
    edges = cv2.Canny((img_cropped_resized * 255).astype(np.uint8), 50, 150)  # Adjusted from (100, 200)
    
    # Thresholding (binary segmentation with Otsu's method for adaptability)
    _, img_thresholded = cv2.threshold((img_cropped_resized * 255).astype(np.uint8), 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the thresholded image as the final output (or change to img_cropped_resized if preferred)
    img_output = img_thresholded / 255.0
    cv2.imwrite(output_path, (img_output * 255).astype(np.uint8))
    
    return {
        'original': img_resized / 255.0,
        'enhanced': img_final,
        'cropped': img_cropped_resized,
        'edges': edges / 255.0,
        'thresholded': img_output
    }

# Process all images in the directory
for idx, filename in enumerate(os.listdir(input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Case-insensitive check
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"preprocessed_{filename}")
        
        # Preprocess without displaying
        result = preprocess_image(image_path, output_path)
        if result is None:
            continue
        
        print(f"Processed {filename} ({idx + 1}/{len(os.listdir(input_dir))})")

print("Preprocessing completed!")