import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the saved PCA model
pca = joblib.load("pca_model.joblib")

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot individual and cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(explained_variance, marker='o', label="Individual Variance")
plt.plot(cumulative_variance, marker='s', label="Cumulative Variance")
plt.axhline(y=0.95, color='r', linestyle='--', label="95% Threshold")

plt.title("PCA - Explained Variance per Principal Component")
plt.xlabel("Principal Component Index")
plt.ylabel("Variance Explained")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
