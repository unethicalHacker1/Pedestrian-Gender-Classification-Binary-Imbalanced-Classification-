# Pedestrian Gender Classification (Binary Imbalanced Classification)

This project implements a pedestrian gender classification pipeline using both **low-level features** (HOG, LBP, GLCM) and **high-level deep features** (FC7 layer from VGG19). The classification task is performed using **Linear SVM** with 10-fold cross-validation. 

## 🔍 Problem Statement

The task is to classify pedestrians as **Male** or **Female** from grayscale and RGB images using classical computer vision and deep learning features. The dataset is **imbalanced** and requires preprocessing and augmentation.

---

## 📁 Dataset

- Source: Custom pedestrian image dataset
- Structure:
MIT-IB/
├── male/
└── female/


## 🧪 Pipeline Overview

### Step 1: Data Preparation
- **Augmentation**: Rotation & horizontal flipping (for females).
- **Preprocessing**: CLAHE applied to grayscale & RGB images.

### Step 2: Feature Engineering
- **Low-Level**:  
- HOG (shape & edges)  
- LBP (texture)  
- GLCM (co-occurrence patterns)
- **High-Level**:
- FC7 Layer features extracted using pretrained **VGG19** on RGB images.

### Step 3: Feature Fusion & Dimensionality Reduction
- **Fusion**: Serial concatenation of low- and high-level features.
- **PCA**: Retained 95% variance (~922 components).

### Step 4: Classification
- **Model**: Linear SVM  
- **Validation**: 10-fold Stratified Cross-Validation  
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---

## 📊 Results

| Metric     | Score   |
|------------|---------|
| Accuracy   | 82.78%  |
| Precision  | 82.95%  |
| Recall     | 81.50%  |
| F1-score   | 82.22%  |

### Confusion Matrix
       Predicted
      | Male | Female
      Actual ---|------|-------
Male | 504 | 96
Female | 106 | 467



---

## 📈 Visualizations

- PCA Explained Variance Plot
- HOG Feature Visualization
- LBP Histogram
- GLCM Matrix
- Confusion Matrix

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: OpenCV, scikit-learn, scikit-image, Keras, TensorFlow, Seaborn, Matplotlib, NumPy

---

## 📂 Folder Structure

├── MIT-IB/
│ ├── male/
│ ├── female/
│ ├── preprocessed/
│ └── preprocessed_rgb/
├── Step_1_Data Augmentation.py
├── Step_2_Pre-Processing(Greyscale).py
├── Step_3_Pre-Processing(rgb).py
├── Step_4_Feature engineering(low-level-LBP,HOG,GLCM).py
├── Step_5_Feature Engineering(High-Level_VGG19).py
├── Step_6_Feature_Fusion_and_PCA.py
├── Step_6b_PCA_Explained_Variance.py
├── Step_7_SVM_Classification.py
├── scaler.joblib
├── pca_model.joblib
├── *.npy (Feature + Label arrays)


## 📌 Contributors

- Muhammad Soban

## 📝 License

This project is for academic use only.
