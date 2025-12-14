import kagglehub
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("shanks0465/braille-character-dataset")
print("Path to dataset files:", path)

# Load images and labels
print("\nLoading images...")
images = []
labels = []

# Get all image files
image_extensions = ['.png', '.jpg', '.jpeg']
image_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(root, file))

print(f"Found {len(image_files)} images")

# Process each image
for img_path in image_files:
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Ensure 28x28
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Extract label from filename (first character before underscore or number)
        filename = os.path.basename(img_path)
        # Get the first character (alphabet letter)
        label = filename[0].upper()
        
        images.append(img_array)
        labels.append(label)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

print(f"\nDataset shape: {X.shape}")
print(f"Number of samples: {len(X)}")
print(f"Number of unique labels: {len(np.unique(y))}")
print(f"Labels: {sorted(np.unique(y))}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded classes: {le.classes_}")

# Reshape X to include channel dimension (28, 28, 1)
X = X.reshape(X.shape[0], 28, 28, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build a simple CNN model
print("\nBuilding CNN model...")
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(26, activation='softmax')  # 26 classes (A-Z)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix - Braille Character Recognition', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history saved as 'training_history.png'")
plt.show()

print("\nModel training completed!")

