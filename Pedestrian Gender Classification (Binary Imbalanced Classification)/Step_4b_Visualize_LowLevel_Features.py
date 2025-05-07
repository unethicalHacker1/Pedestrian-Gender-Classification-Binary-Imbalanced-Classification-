import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure
import seaborn as sns

# Sample grayscale image path
sample_image_path = 'MIT-IB/preprocessed/female/pre_00010_female_fore.jpg'

# Load and resize
img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 256))

# ------------------- HOG -------------------
features_hog, hog_image = hog(img,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True)

hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.figure(figsize=(6, 4))
plt.imshow(hog_image, cmap='gray')
plt.title("HOG Feature Visualization")
plt.axis('off')
plt.tight_layout()
plt.show()

# ------------------- LBP -------------------
lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

plt.figure(figsize=(6, 4))
plt.bar(range(9), hist_lbp[:9])
plt.title("LBP Histogram (Uniform Patterns)")
plt.xlabel("Pattern Index")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ------------------- GLCM -------------------
glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
glcm_matrix = glcm[:, :, 0, 0]

plt.figure(figsize=(6, 5))
sns.heatmap(glcm_matrix, cmap='Greys', cbar=True)
plt.title("GLCM Co-occurrence Matrix (0°)")
plt.xlabel("Gray Level j")
plt.ylabel("Gray Level i")
plt.tight_layout()
plt.show()
