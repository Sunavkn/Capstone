import cv2 
import albumentations as A 
import os 
 
# Define input and output paths 
input_path = r'D:\\Capstone\\image_augmentation\\code\\xray.jpg' 
output_folder = r'D:\\Capstone\\image_augmentation\\output' 
 
# Ensure output directory exists 
os.makedirs(output_folder, exist_ok=True) 
 
# Load the input chest X-ray image 
image = cv2.imread(input_path) 
 
if image is None: 
    print(f"Error: Could not load {input_path}. Check if the file exists.") 
    exit() 
 
# Define 10 augmentation pipelines
augmentations = [ 
    A.Compose([ 
        A.Rotate(limit=5, p=1), 
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1), 
        A.GaussianBlur(blur_limit=3, p=1)
    ]), 
    A.Compose([ 
        A.Affine(shear=5, p=1),   
        A.Affine(scale=(1.1, 1.1), p=1) 
    ]), 
    A.Compose([ 
        A.Affine(scale=(0.9, 0.9), p=1), 
        A.GaussianBlur(blur_limit=3, p=1)
    ]), 
    A.Compose([ 
        A.Rotate(limit=-5, p=1), 
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1) 
    ]), 
    A.Compose([ 
        A.RandomBrightnessContrast(brightness_limit=-0.1, contrast_limit=-0.2, p=1) 
    ]), 
    A.Compose([ 
        A.Affine(shear=5, p=1),   
        A.Rotate(limit=5, p=1) 
    ]), 
    A.Compose([ 
        A.Affine(scale=(1.1, 1.1), p=1), 
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1) 
    ]), 
    A.Compose([ 
        A.Affine(scale=(0.9, 0.9), p=1), 
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1) 
    ]), 
    A.Compose([ 
        A.Rotate(limit=5, p=1), 
        A.GaussianBlur(blur_limit=3, p=1)
    ]), 
    A.Compose([ 
        A.Affine(shear=5, p=1),  
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1), 
        A.GaussianBlur(blur_limit=3, p=1)
    ]) 
] 
 
# Apply each augmentation pipeline and save the result 
for i, aug in enumerate(augmentations, 1): 
    augmented = aug(image=image)['image'] 
    output_path = os.path.join(output_folder, f'augmented_albumentations_{i}.jpg') 
    cv2.imwrite(output_path, augmented) 
 
print(f"✅ 10 augmented images have been saved in '{output_folder}'.")