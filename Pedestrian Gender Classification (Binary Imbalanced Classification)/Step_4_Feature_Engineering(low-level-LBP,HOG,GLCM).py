import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix , graycoprops

# Paths to preprocessed grayscale images
female_dir = 'MIT-IB/preprocessed/female/'
male_dir = 'MIT-IB/preprocessed/male/'

X_combined = []
y_combined = []

# HOG parameters
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'feature_vector': True
}

# LBP parameters
lbp_radius = 1
lbp_n_points = 8 * lbp_radius

# GLCM parameters
glcm_distances = [1]
glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def extract_all_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 256))  # Resize for HOG consistency

    # HOG
    hog_features = hog(img, **hog_params)

    # LBP
    lbp = local_binary_pattern(img, lbp_n_points, lbp_radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_n_points + 3), density=True)

    # GLCM
    glcm = graycomatrix (img, distances=glcm_distances, angles=glcm_angles, levels=256, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        values = graycoprops(glcm, prop).flatten()
        glcm_features.extend(values)

    # Combine all features
    combined = np.concatenate([hog_features, lbp_hist, glcm_features])
    return combined

# Helper to process a folder
def process_folder(folder_path, label):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            features = extract_all_features(img_path)
            X_combined.append(features)
            y_combined.append(label)
        except Exception as e:
            print(f"⚠️ Skipping: {img_path} → {e}")


# Run extraction
print("Extracting features from female images...")
process_folder(female_dir, label=1)

print("Extracting features from male images...")
process_folder(male_dir, label=0)

# Convert to NumPy arrays
X_combined = np.array(X_combined)
y_combined = np.array(y_combined)

print("Feature extraction complete.")
print(f"Feature vector shape: {X_combined.shape}")
print(f"Female samples: {np.sum(y_combined == 1)}")
print(f"Male samples: {np.sum(y_combined == 0)}")

# Save features and labels
np.save("X_combined.npy", X_combined)
np.save("y_combined.npy", y_combined)

print("Features saved as 'X_combined.npy' and 'y_combined.npy'")
