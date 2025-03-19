import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Define constants
IMG_SIZE = (224, 224)  # Same size used during training
CLASS_NAMES = ['Normal', 'Tuberculosis']  # Add 'Silicosis' when you include it later

# Load the model
def load_model(model_path):
    """Load the saved model"""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

# Preprocess an image
def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    # Read the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
    # Resize to the required size
    img = tf.image.resize(img, IMG_SIZE)
    
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    # Handle grayscale images (convert to RGB)
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    
    return img

# Predict on a single image
def predict_single_image(model, image_path):
    """Make a prediction on a single image"""
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    
    return {
        'class_idx': predicted_class_idx,
        'class_name': CLASS_NAMES[predicted_class_idx],
        'confidence': confidence,
        'all_probabilities': prediction[0]
    }

# Visualize the prediction
def visualize_prediction(image_path, prediction_result):
    """Visualize the image with prediction result"""
    # Read the image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    img = cv2.resize(img, IMG_SIZE)
    
    # Create a figure for visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    
    # Create title with prediction info
    title = f"Prediction: {prediction_result['class_name']}\n"
    title += f"Confidence: {prediction_result['confidence']:.2f}%"
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print detailed prediction
    print(f"Prediction: {prediction_result['class_name']}")
    print(f"Confidence: {prediction_result['confidence']:.2f}%")
    print("\nProbabilities for all classes:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {prediction_result['all_probabilities'][i] * 100:.2f}%")

# Batch prediction on a directory of images
def predict_directory(model, directory_path):
    """Make predictions on all images in a directory"""
    results = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            result = predict_single_image(model, image_path)
            results.append({
                'filename': filename,
                'result': result
            })
            print(f"Predicted {filename} as {result['class_name']} with {result['confidence']:.2f}% confidence")
    
    return results

# Main function
def main():
    # Path to your saved model
    model_path = "lung_xray_classifier_efficientnet_final.h5"
    
    # Load the model
    model = load_model(model_path)
    
    # Example 1: Predict on a single image
    image_path = input("Enter the path to an X-ray image (or press Enter to skip): ")
    if image_path:
        result = predict_single_image(model, image_path)
        visualize_prediction(image_path, result)
    
    # Example 2: Predict on a directory of images
    directory_path = input("Enter the path to a directory with X-ray images (or press Enter to skip): ")
    if directory_path:
        results = predict_directory(model, directory_path)
        
        # Print summary
        print("\n===== Summary of Predictions =====")
        for class_name in CLASS_NAMES:
            count = sum(1 for r in results if r['result']['class_name'] == class_name)
            print(f"{class_name}: {count} images")
        print("=================================")

if __name__ == "__main__":
    main()