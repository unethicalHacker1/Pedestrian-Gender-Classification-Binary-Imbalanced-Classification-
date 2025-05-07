import os
import cv2

# Input Paths
original_female_dir = 'MIT-IB/female/'
augmented_female_dir = 'MIT-IB/female_aug/'
original_male_dir = 'MIT-IB/male/'

# Output Paths
output_female_dir = 'MIT-IB/preprocessed_rgb/female/'
output_male_dir = 'MIT-IB/preprocessed_rgb/male/'

# Create output directories if not present
os.makedirs(output_female_dir, exist_ok=True)
os.makedirs(output_male_dir, exist_ok=True)

# --- CLAHE + Resize Function ---
def preprocess_rgb_image(image_path):
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        print(f"Skipping invalid image: {image_path}")
        return None

    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge and convert back to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe

# Process and save images
def process_images(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        processed = preprocess_rgb_image(img_path)
        if processed is not None:
            save_path = os.path.join(output_dir, f"pre_rgb_{img_name}")
            cv2.imwrite(save_path, processed)

# Process Female Images (original + augmented)
print("Preprocessing original female images...")
process_images(original_female_dir, output_female_dir)

print("Preprocessing augmented female images...")
process_images(augmented_female_dir, output_female_dir)

# Process Male Images
print("Preprocessing original male images...")
process_images(original_male_dir, output_male_dir)

print("RGB preprocessing complete and saved in 'MIT-IB/preprocessed_rgb/'")
