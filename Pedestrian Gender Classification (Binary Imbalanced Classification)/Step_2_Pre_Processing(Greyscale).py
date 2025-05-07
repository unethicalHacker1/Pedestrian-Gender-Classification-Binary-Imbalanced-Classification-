import cv2
import os

# Dataset Paths
original_female_dir = 'MIT-IB/female/'
augmented_female_dir = 'MIT-IB/female_aug/'
original_male_dir = 'MIT-IB/male/'

# Save preprocessed images here
output_female_dir = 'MIT-IB/preprocessed/female/'
output_male_dir = 'MIT-IB/preprocessed/male/'

# Create output folders if they don't exist
os.makedirs(output_female_dir, exist_ok=True)
os.makedirs(output_male_dir, exist_ok=True)

# Store arrays and labels
preprocessed_images = []
labels = []

# Preprocessing with Gaussian Blur + CLAHE
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        print(f"Skipping invalid image: {image_path}")
        return None
    
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced

# Process and Save Images
def process_and_save_images(input_dir, output_dir, label_value):
    for img_name in os.listdir(input_dir):
        path = os.path.join(input_dir, img_name)
        img = preprocess_image(path)
        if img is None:
            continue  # Skip invalid images
        preprocessed_images.append(img)
        labels.append(label_value)

        # Save the image
        save_path = os.path.join(output_dir, f"pre_{img_name}")
        cv2.imwrite(save_path, img)

# Process Female Original + Augmented
print("Preprocessing original female images...")
process_and_save_images(original_female_dir, output_female_dir, 1)

print("Preprocessing augmented female images...")
process_and_save_images(augmented_female_dir, output_female_dir, 1)

# Process Male Images
print("Preprocessing original male images...")
process_and_save_images(original_male_dir, output_male_dir, 0)

# Summary
print(f"Total preprocessed images: {len(preprocessed_images)}")
print(f"Female samples: {labels.count(1)}")
print(f"Male samples: {labels.count(0)}")
print("Preprocessing completed.")
