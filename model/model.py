import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (224, 224)  # EfficientNet preferred size
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_CLASSES = 2  # Normal and TB (scalable to 3 for silicosis)
MODEL_NAME = "lung_xray_classifier_efficientnet"

# Paths to your preprocessed image directories 
# Note: Update these paths to match your actual directories
NORMAL_DIR = "C:/Users/jayan/SEM 6/Capstone/TB_Chest_Radiography_Database/normal_preprocess_1"
TB_DIR = "C:/Users/jayan/SEM 6/Capstone/TB_Chest_Radiography_Database/tb_preprocess_1"
# SILICOSIS_DIR = "path/to/silicosis/images"  # Uncomment when adding silicosis

# Function to create dataset from multiple directories
def create_dataset(image_dirs, class_names):
    """
    Creates a dataset from multiple directories and assigns class labels
    """
    file_paths = []
    labels = []
    
    for class_idx, (image_dir, class_name) in enumerate(zip(image_dirs, class_names)):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(image_dir, filename))
                labels.append(class_idx)
    
    # Convert to numpy arrays
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Function to load and preprocess images
def load_image(file_path, label):
    """
    Load and preprocess image from file path
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    
    # Handle grayscale images - convert to RGB
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    
    return img, label

# Create TF dataset
def create_tf_dataset(file_paths, labels, batch_size, augment=False):
    """
    Create a TensorFlow dataset from file paths and labels
    """
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        # Add data augmentation for training
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y), 
                             num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Build the model
def build_model(num_classes):
    """
    Build an EfficientNetB2-based model for chest X-ray classification
    """
    # Load pre-trained EfficientNetB2 without top layer
    base_model = EfficientNetB2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze the base model (for initial training)
    base_model.trainable = False
    
    # Create model structure
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# Function to train the model
def train_model(model, train_ds, val_ds, epochs):
    """
    Train the model with callbacks for early stopping and checkpointing
    """
    # Create callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        f"{MODEL_NAME}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )
    
    return history

# Function to unfreeze and fine-tune the model
def fine_tune_model(model, base_model, train_ds, val_ds, fine_tune_epochs=10):
    """
    Unfreeze the base model and fine-tune with a lower learning rate
    """
    # Unfreeze the base_model
    base_model.trainable = True
    
    # Freeze earlier layers (first 100 layers)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile model with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks for fine-tuning
    checkpoint_cb = callbacks.ModelCheckpoint(
        f"{MODEL_NAME}_finetuned.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Fine-tune the model
    history_fine = model.fit(
        train_ds,
        epochs=fine_tune_epochs,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    return history_fine

# Function to plot training history
def plot_training_history(history, fine_tune_history=None):
    """
    Plot the training and validation accuracy/loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # If fine-tuning history is provided, append to the history
    if fine_tune_history is not None:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_training_history.png")
    plt.show()

# Function to evaluate the model
def evaluate_model(model, test_ds, class_names):
    """
    Evaluate the model and generate a confusion matrix and classification report
    """
    # Get predictions
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels from the dataset
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    
    # Generate a classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Generate a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_confusion_matrix.png")
    plt.show()
    
    return report_df

# Function to make prediction on a single image
def predict_image(model, image_path, class_names):
    """
    Make a prediction on a single image
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Handle grayscale images
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    print(f"Predicted class: {class_names[predicted_class]} with {confidence:.2f}% confidence")
    return predicted_class, confidence

# Main execution
def main():
    # Define class names
    class_names = ['Normal', 'Tuberculosis']  # Add 'Silicosis' when available
    image_dirs = [NORMAL_DIR, TB_DIR]  # Add SILICOSIS_DIR when available
    
    print("Creating dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = create_dataset(image_dirs, class_names)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create TensorFlow datasets
    train_ds = create_tf_dataset(X_train, y_train, BATCH_SIZE, augment=True)
    val_ds = create_tf_dataset(X_val, y_val, BATCH_SIZE)
    test_ds = create_tf_dataset(X_test, y_test, BATCH_SIZE)
    
    # Build model
    print("Building model...")
    model, base_model = build_model(NUM_CLASSES)
    model.summary()
    
    # Initial training with frozen base model
    print("Training initial model...")
    history = train_model(model, train_ds, val_ds, EPOCHS)
    
    # Fine-tuning
    print("Fine-tuning model...")
    fine_tune_history = fine_tune_model(model, base_model, train_ds, val_ds)
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_model(model, test_ds, class_names)
    print(eval_results)
    
    # Save the final model
    model.save(f"{MODEL_NAME}_final.h5")
    print(f"Model saved as {MODEL_NAME}_final.h5")
    
    # Test on a sample image from each class
    print("Sample predictions:")
    if len(X_test) > 0:
        normal_sample = X_test[y_test == 0][0] if any(y_test == 0) else None
        tb_sample = X_test[y_test == 1][0] if any(y_test == 1) else None
        
        if normal_sample:
            print("Normal sample prediction:")
            predict_image(model, normal_sample, class_names)
        
        if tb_sample:
            print("TB sample prediction:")
            predict_image(model, tb_sample, class_names)

# Run the main function
if __name__ == "__main__":
    main()