import cv2
import numpy as np

# Load the input chest X-ray image
image = cv2.imread(r'D:\\Capstone\\image_augmentation\\xray.jpg')

# Define augmentation functions (unchanged from your code)
def shear_x(image, angle):
    """Shear the image along the x-axis by a specified angle in degrees."""
    rows, cols = image.shape[:2]
    M = np.float32([[1, np.tan(np.deg2rad(angle)), 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

def rotate(image, angle):
    """Rotate the image by a specified angle in degrees."""
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def zoom_in(image, scale):
    """Zoom into the image by cropping a central region and resizing to original size."""
    rows, cols = image.shape[:2]
    crop_size_rows = int(rows / scale)
    crop_size_cols = int(cols / scale)
    start_row = (rows - crop_size_rows) // 2
    start_col = (cols - crop_size_cols) // 2
    cropped = image[start_row:start_row + crop_size_rows, start_col:start_col + crop_size_cols]
    return cv2.resize(cropped, (cols, rows))

def zoom_out(image, scale):
    """Zoom out by shrinking the image and padding the borders."""
    rows, cols = image.shape[:2]
    new_rows = int(rows * scale)
    new_cols = int(cols * scale)
    resized = cv2.resize(image, (new_cols, new_rows))
    pad_top = (rows - new_rows) // 2
    pad_bottom = rows - new_rows - pad_top
    pad_left = (cols - new_cols) // 2
    pad_right = cols - new_cols - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                             cv2.BORDER_CONSTANT, value=0)

def adjust_brightness(image, beta):
    """Adjust brightness by adding/subtracting a constant value."""
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)

def adjust_contrast(image, alpha):
    """Adjust contrast by scaling pixel values around the mean."""
    mean = np.mean(image)
    beta = mean * (1 - alpha)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image, mean=0, std=10):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Apply 10 different combinations of augmentations
aug1 = add_gaussian_noise(adjust_brightness(rotate(image, 5), 25))  # Rotate + Brightness + Noise
aug2 = zoom_in(shear_x(image, 5), 1.1)                             # Shear + Zoom In
aug3 = add_gaussian_noise(zoom_out(image, 0.9))                    # Zoom Out + Noise
aug4 = adjust_contrast(rotate(image, -5), 1.2)                     # Rotate + Contrast
aug5 = adjust_contrast(adjust_brightness(image, -25), 0.8)         # Brightness + Contrast
aug6 = rotate(shear_x(image, 5), 5)                                # Shear + Rotate
aug7 = adjust_brightness(zoom_in(image, 1.1), 25)                  # Zoom In + Brightness
aug8 = adjust_contrast(zoom_out(image, 0.9), 1.2)                  # Zoom Out + Contrast
aug9 = add_gaussian_noise(rotate(image, 5))                        # Rotate + Noise
aug10 = add_gaussian_noise(adjust_brightness(shear_x(image, 5), 25))  # Shear + Brightness + Noise

# Save the augmented images
augmented_images = [aug1, aug2, aug3, aug4, aug5, aug6, aug7, aug8, aug9, aug10]
for i, aug in enumerate(augmented_images, 1):
    cv2.imwrite(f'augmented_combined_{i}.jpg', aug)

print("10 augmented images with combined transformations have been generated and saved as 'augmented_combined_1.jpg' to 'augmented_combined_10.jpg'.")