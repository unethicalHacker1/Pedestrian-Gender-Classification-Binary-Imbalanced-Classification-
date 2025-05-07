import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load features and labels
X_low = np.load("X_combined.npy")
X_high = np.load("X_vgg_fc7.npy")
y_low = np.load("y_combined.npy")
y_high = np.load("y_vgg_fc7.npy")

# Sanity check
assert X_low.shape[0] == X_high.shape[0], "Feature count mismatch!"
assert np.array_equal(y_low, y_high), "Label mismatch!"

# Step 2: Feature Fusion (serial-based: horizontal stack)
X_fused = np.hstack((X_low, X_high))
print(f"Fused feature shape: {X_fused.shape}")

# Step 3: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fused)

# Step 4: Apply PCA (retain 95% variance)
pca = PCA(n_components=0.95, svd_solver='full')
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced feature shape after PCA: {X_reduced.shape}")
print(f"PCA retained {pca.n_components_} components")

# Step 5: Save outputs
np.save("X_fused_pca.npy", X_reduced)
np.save("y_fused_pca.npy", y_low)
joblib.dump(pca, "pca_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Feature fusion and PCA completed successfully.")
