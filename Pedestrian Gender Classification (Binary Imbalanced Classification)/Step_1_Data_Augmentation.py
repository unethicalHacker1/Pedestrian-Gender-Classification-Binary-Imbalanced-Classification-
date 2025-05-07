import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
female_dir = 'MIT-IB/female/'
augmented_dir = 'MIT-IB/female_aug/'

# Create directories if they don't exist
os.makedirs(augmented_dir, exist_ok=True)


# Step 1: Augment female images
datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True) # Data augmentation parameters(flip, rotation, etc.)

print("Augmenting images...")
for img_name in os.listdir(female_dir):
    img_path = os.path.join(female_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    x = img[np.newaxis, ...]

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir,
                               save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 1:  # Generate only 1 augmented image per original
            break
print("Augmentation completed.")

print(f"Combined dataset saved to: {augmented_dir}")
