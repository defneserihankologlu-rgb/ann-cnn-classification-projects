import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")
print("Path to dataset files:", path)

# Load the dataset
import os
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    csv_path = os.path.join(path, csv_files[0])
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
else:
    print("No CSV file found in the dataset path")
    exit()

# Explore the dataset
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nTarget variable distribution:")
print(df.iloc[:, -1].value_counts())
print("\nDataset info:")
print(df.info())

# Prepare the data
# Try to find the target column (usually named 'fetal_health' or last column)
target_col = None
if 'fetal_health' in df.columns:
    target_col = 'fetal_health'
elif 'target' in df.columns:
    target_col = 'target'
else:
    # Use last column as target
    target_col = df.columns[-1]

print(f"\nUsing '{target_col}' as target variable")

X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Encode target variable if it's not numeric
if y.dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Encoded classes: {le.classes_}")
else:
    # If numeric, make sure it's 0-indexed (0, 1, 2 for 3 classes)
    unique_classes = np.unique(y)
    if len(unique_classes) == 3 and min(unique_classes) != 0:
        # Map to 0, 1, 2 if needed
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        y = np.array([class_mapping[val] for val in y])
        print(f"Mapped classes: {class_mapping}")

print(f"\nNumber of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple ANN model
print("\nBuilding ANN model...")
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # 3 classes
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
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nModel training completed!")

