import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import torch
import joblib

# Paths to saved models and input image
svm_model_file = '/Users/Sunav/cprog/sem3_workspace/college/capstone/svm_model.pkl'
pca_model_file = '/Users/Sunav/cprog/sem3_workspace/college/capstone/pca_model.pkl'
image_path = '/Users/Sunav/cprog/sem3_workspace/college/capstone/Screenshot 2025-03-20 at 1.24.52 PM.png'  # Replace with your test image path
image_size = 224

# Check device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ViT
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)

# Load saved models
svm = joblib.load(svm_model_file)
pca = joblib.load(pca_model_file)
print("Loaded SVM and PCA models")

# Function to extract ViT features
def extract_features(image):
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].cpu().numpy()

# Load and preprocess the input image
img = Image.open(image_path).convert('RGB')
img = img.resize((image_size, image_size))
img_array = np.array(img)

# Extract features
features = extract_features(img_array)

# Apply PCA
pca_features = pca.transform(features)

# Predict with confidence
prediction = svm.predict(pca_features)
confidence = svm.predict_proba(pca_features)[0]  # Probability for each class

# Interpret results
class_names = ['Normal', 'TB']
predicted_class = class_names[prediction[0]]
confidence_normal = confidence[0]
confidence_tb = confidence[1]

print(f"Predicted class: {predicted_class}")
print(f"Confidence - Normal: {confidence_normal:.4f}")
print(f"Confidence - TB: {confidence_tb:.4f}")