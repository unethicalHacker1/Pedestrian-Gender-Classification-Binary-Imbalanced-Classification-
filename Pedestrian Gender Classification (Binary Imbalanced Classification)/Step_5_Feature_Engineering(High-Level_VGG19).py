import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Dataset Paths ---
female_dir = 'MIT-IB/preprocessed_rgb/female/'
male_dir = 'MIT-IB/preprocessed_rgb/male/'

# --- Load VGG19 model and extract from FC2 (also known as FC7) ---
base_model = VGG19(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# --- Output Lists ---
X_vgg = []
y_vgg = []

# --- Function to Extract Deep Features ---
def extract_vgg19_features(image_path, label):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping: {image_path}")
        return

    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = model.predict(img, verbose=0)
    X_vgg.append(features.flatten())
    y_vgg.append(label)

# --- Extract from Female Images ---
print("Extracting features from female images...")
for img_name in os.listdir(female_dir):
    extract_vgg19_features(os.path.join(female_dir, img_name), label=1)

# --- Extract from Male Images ---
print("Extracting features from male images...")
for img_name in os.listdir(male_dir):
    extract_vgg19_features(os.path.join(male_dir, img_name), label=0)

# --- Convert to NumPy Arrays ---
X_vgg = np.array(X_vgg)
y_vgg = np.array(y_vgg)

# --- Save to disk (optional but recommended) ---
np.save("X_vgg_fc7.npy", X_vgg)
np.save("y_vgg_fc7.npy", y_vgg)

# --- Summary ---
print("VGG19 High-level feature extraction complete.")
print(f"Feature vector shape: {X_vgg.shape}")
print(f"Female samples: {np.sum(y_vgg == 1)}")
print(f"Male samples: {np.sum(y_vgg == 0)}")
