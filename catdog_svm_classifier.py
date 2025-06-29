import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configuration
DATADIR = "./PetImages"
CATEGORIES = ["Cat", "Dog"]
IMG_SIZE = 64
SAMPLES_PER_CLASS = 1000  # Limit for speed

# Load and preprocess data
data = []
print("\n📦 Loading and processing images with HOG features...")
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    label = CATEGORIES.index(category)

    for img in tqdm(os.listdir(path)[:SAMPLES_PER_CLASS]):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                continue
            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # Extract HOG features
            features = hog(resized,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           orientations=9,
                           block_norm='L2-Hys')
            data.append([features, label])
        except Exception:
            continue

print(f"\n✅ Total images loaded: {len(data)}")

if len(data) == 0:
    print("❌ No images loaded. Check folder structure and files.")
    exit()

# Shuffle and split
random.shuffle(data)
X = np.array([features for features, label in data])
y = np.array([label for features, label in data])

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model with RBF kernel
print("\n🤖 Training SVM model with RBF kernel...")
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy with HOG + SVM: {accuracy:.2f}")

# Show sample predictions
print("\n🖼️ Sample predictions (labels only, image not shown due to HOG transformation):")
for i in range(5):
    print(f"Predicted: {CATEGORIES[y_pred[i]]}")
