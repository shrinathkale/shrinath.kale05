import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load the dataset
# Make sure to specify the correct path to your dataset
dataset_path = 'dataset'  # Change to your dataset path
categories = os.listdir(dataset_path)

# Load images and labels
images = []
labels = []
for category in categories:
    category_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Assign label based on folder index
    for img in os.listdir(category_path):
        img_path = os.path.join(category_path, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (128, 128))  # Resize images
        images.append(img_array)
        labels.append(label)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Create a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Use softmax if more than 2 classes
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Step 6: Make predictions
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return "Tumor" if prediction[0][0] > 0.5 else "No Tumor"

# Example prediction
test_image_path = 'pred16.jpg'  # Change to your test image path
result = predict_image(test_image_path)
print(f'Prediction: {result}')
