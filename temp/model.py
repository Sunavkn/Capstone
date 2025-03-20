import os
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import time
import joblib  # For saving models

# Dataset path
data_dir = '/Users/Sunav/cprog/sem3_workspace/college/capstone/dataset'
image_size = 224
batch_size = 32
feature_file = 'features.npy'
svm_model_file = 'svm_model.pkl'
pca_model_file = 'pca_model.pkl'

# Check device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

start_time = time.time()

# Load images and labels
images = []
labels = []
for label in ['TB', 'Normal']:
    path = os.path.join(data_dir, label)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Folder {path} does not exist!")
    img_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(img_files)} images in {label}")
    for img_name in img_files:
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((image_size, image_size))
        images.append(np.array(img))
        labels.append(1 if label == 'TB' else 0)

images = np.array(images)
labels = np.array(labels)
print(f"Loaded {len(images)} images in {time.time() - start_time:.2f} seconds")
print(f"Label distribution: TB={np.sum(labels == 1)}, Normal={np.sum(labels == 0)}")

# Load ViT
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)

# Batch feature extraction
def extract_features_batch(image_batch):
    inputs = processor(images=list(image_batch), return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].cpu().numpy()

print("Starting feature extraction...")
features = []
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    batch_features = extract_features_batch(batch)
    features.extend(batch_features)
    if (i + batch_size) % 1000 < batch_size:
        np.save(feature_file, np.array(features))
        print(f"Saved features for {min(i + batch_size, len(images))}/{len(images)} images")
np.save(feature_file, np.array(features))
print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

features = np.load(feature_file)
print(f"Loaded features: {features.shape}")

# PCA
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(features)
print(f"PCA completed in {time.time() - start_time:.2f} seconds")

# Split data
X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.2, random_state=42)
print(f"Training set: TB={np.sum(y_train == 1)}, Normal={np.sum(y_train == 0)}")
print(f"Test set: TB={np.sum(y_test == 1)}, Normal={np.sum(y_test == 0)}")

# Train SVM
svm = SVC(kernel='linear', C=1, probability=True)  # Enable probability for confidence scores
svm.fit(X_train, y_train)
print(f"SVM training completed in {time.time() - start_time:.2f} seconds")

# Evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.4f}')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'TB']))

# Save models
joblib.dump(svm, svm_model_file)
joblib.dump(pca, pca_model_file)
print(f"Saved SVM model to {svm_model_file}")
print(f"Saved PCA model to {pca_model_file}")

# Optional: Cross-validation (uncomment to run)
# scores = cross_val_score(svm, pca_features, labels, cv=5, scoring='accuracy')
# print(f"5-Fold CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print(f"Total runtime: {time.time() - start_time:.2f} seconds")