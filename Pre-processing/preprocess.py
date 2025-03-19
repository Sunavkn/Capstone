import cv2
import numpy as np
import os
from skimage import exposure
import matplotlib.pyplot as plt

# Define input and output directories
input_dir = r"C:\Users\Hp\Documents\Capstone\Dataset\New folder"  # Replace with your folder path
output_dir = r"C:\Users\Hp\Documents\Capstone\Dataset\Preprocessed"  # Replace with your output folder
os.makedirs(output_dir, exist_ok=True)

# Parameters
IMG_SIZE = (256, 256)  # Standard size for resizing
CROP_SIZE = 5  # Pixels to crop from each side (adjust based on your X-rays)

# Function to preprocess a single image
def preprocess_image(image_path, output_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    
    # Resize image
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Noise reduction (Gaussian blur)
    img_blurred = cv2.GaussianBlur(img_normalized, (5, 5), 0)
    
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply((img_blurred * 255).astype(np.uint8))  # Convert back to 0-255 range
    
    # Normalize again after enhancement
    img_final = img_enhanced / 255.0
    
    # Cropping (remove borders, adjust CROP_SIZE as needed)
    h, w = img_final.shape
    img_cropped = img_final[CROP_SIZE:h-CROP_SIZE, CROP_SIZE:w-CROP_SIZE]
    
    # Resize back to IMG_SIZE after cropping (if needed)
    img_cropped_resized = cv2.resize(img_cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Edge detection (Canny)
    edges = cv2.Canny((img_cropped_resized * 255).astype(np.uint8), 100, 200)
    
    # Thresholding (binary segmentation)
    _, img_thresholded = cv2.threshold((img_cropped_resized * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    # Combine preprocessed images (you can choose which to save or use)
    # Here, we save the thresholded image as the final output; adjust as needed
    img_output = img_thresholded / 255.0  # Normalize back to [0, 1]
    
    # Save preprocessed image
    cv2.imwrite(output_path, (img_output * 255).astype(np.uint8))  # Save as 0-255 range
    
    return {
        'original': img_resized / 255.0,
        'enhanced': img_final,
        'cropped': img_cropped_resized,
        'edges': edges / 255.0,
        'thresholded': img_output
    }

# Process all images in the directory
for filename in os.listdir(input_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Adjust extensions as needed
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"preprocessed_{filename}")
        
        # Preprocess and get results
        preprocessed_imgs = preprocess_image(image_path, output_path)
        
        # Optional: Visualize all steps
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(preprocessed_imgs['original'], cmap='gray')
        plt.title("Original (Resized)")
        
        plt.subplot(2, 3, 2)
        plt.imshow(preprocessed_imgs['enhanced'], cmap='gray')
        plt.title("Enhanced (CLAHE)")
        
        plt.subplot(2, 3, 3)
        plt.imshow(preprocessed_imgs['cropped'], cmap='gray')
        plt.title("Cropped")
        
        plt.subplot(2, 3, 4)
        plt.imshow(preprocessed_imgs['edges'], cmap='gray')
        plt.title("Edges (Canny)")
        
        plt.subplot(2, 3, 5)
        plt.imshow(preprocessed_imgs['thresholded'], cmap='gray')
        plt.title("Thresholded")
        
        plt.tight_layout()
        plt.show()

print("Preprocessing completed!")