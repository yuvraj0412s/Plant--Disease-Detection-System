# 🌿 Plant Disease Detection System

A deep learning-based web application that detects plant diseases from leaf images using **EfficientNetB4** and **Flask**. It provides accurate predictions across 38 classes and is trained on the PlantVillage dataset.

<p align="center">
  <img src="screenshots/accuracy_loss_graph.png" alt="Training Accuracy vs Loss" width="600"/>
</p>

---

## 🚀 Features

- 📷 Upload leaf images for real-time disease prediction
- 🧠 Deep learning with **EfficientNetB4** using transfer learning
- 🔥 Achieved high accuracy on multiclass classification (38 disease categories)
- 🌐 User-friendly interface with **Flask**
- 🧪 Sample test images included for demonstration

---

## 🧠 Model Architecture

- **Base model**: EfficientNetB4 (pretrained on ImageNet)
- **Top layers**: Global Average Pooling → Dense layers → Softmax
- **Loss function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

Trained on augmented data using:
- Random flip, rotation, zoom
- Rescaling and resizing

---

## 📁 Project Structure

Plant-Disease-Detection-System/
├── app.py # Flask app
├── plant_disease.json # Class label mapping
├── requirements.txt # Dependencies
├── README.md
├── models/
│ └── Plant_Disease_Detection_Model_01.keras # (Excluded in repo)
├── sample_test_images/ # Sample images for testing
│ ├── apple_scab.jpg
│ ├── grape_black_rot.jpg
│ └── tomato_early_blight.jpg
├── screenshots/ # Screenshots of UI and graphs
│ ├── accuracy_loss_graph.png
│ ├── home.png
│ └── result.png
└── docs/
└── dataset_info.md # Optional: extra info about dataset


---

## 🧪 Sample Usage

1. Launch the Flask app:
   ```bash
   python app.py

