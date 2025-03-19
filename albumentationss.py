import cv2 
import albumentations as A 
import os 

# Define input and output paths 
input_folder = r'D:\Capstone\image_augmentation\code\Tuberculosis' 
output_folder = input_folder  # Save in the same folder 

# Ensure output directory exists 
os.makedirs(output_folder, exist_ok=True) 

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

# Process each image in the folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    image = cv2.imread(input_path) 
    
    if image is None: 
        print(f"Error: Could not load {input_path}. Skipping.") 
        continue 
    
    # Apply each augmentation pipeline and save the result 
    name, ext = os.path.splitext(filename)
    for i, aug in enumerate(augmentations, 1): 
        augmented = aug(image=image)['image'] 
        output_path = os.path.join(output_folder, f'{name}.{i}{ext}') 
        cv2.imwrite(output_path, augmented) 

print(f"✅ Augmented images have been saved in '{output_folder}'.")
