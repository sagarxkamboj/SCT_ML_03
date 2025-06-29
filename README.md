# SCT_ML_03
🐱🐶 Cat vs Dog Image Classifier using SVM + HOG
This project implements a machine learning model to classify images of cats and dogs using a Support Vector Machine (SVM) with Histogram of Oriented Gradients (HOG) features.

📌 Features
✅ Uses the popular Kaggle Dogs vs Cats dataset

📷 Preprocesses grayscale images using OpenCV

📈 Extracts meaningful features using HOG

🧠 Trains a Support Vector Machine (SVM) with RBF kernel

📊 Achieves ~74% accuracy without using deep learning

💾 Model saving and reloading (optional)

🖼️ Easily extendable for visual predictions and testing new images

🧰 Technologies Used
Python

OpenCV

scikit-learn

scikit-image (HOG)

tqdm

matplotlib (optional visualization)

📁 Dataset
Source: Kaggle Dogs vs. Cats Dataset

Structure:

Copy
Edit
PetImages/
├── Cat/
└── Dog/
🚀 How to Run
Clone the repo

Place the PetImages dataset in the root directory

Install dependencies:

bash
Copy
Edit
pip install opencv-python scikit-learn scikit-image tqdm matplotlib
Run the classifier:

bash
Copy
Edit
python catdog.py
📈 Accuracy
Achieved approximately 74% accuracy on a validation set of 400 images using HOG + RBF SVM.

📌 Future Improvements
Add prediction on custom uploaded images

Visualize predictions with actual images

Integrate with a simple GUI or Flask app
