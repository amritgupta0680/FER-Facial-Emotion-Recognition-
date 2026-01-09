# 🎭 Facial Emotion Recognition (FER)

A deep learning–based Facial Emotion Recognition system with a Flask web interface that detects human emotions in real time using a webcam.

---

## 📌 Project Overview

This project focuses on recognizing facial emotions from images and live video streams.  
The system is divided into two main parts:

1. Model Training & Experimentation (Jupyter Notebook)
2. Real-Time Web Application (Flask)

The trained deep learning model is integrated into a Flask application to perform real-time emotion detection using a webcam.

---

## 📓 Training & Dataset Details (`FER 2 database.ipynb`)

The notebook **`FER 2 database.ipynb`** contains the complete pipeline for training and evaluating the facial emotion recognition model.

### 🔹 Dataset
- RAF-DB dataset used for fine-tuning
- 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Dataset split into training and validation sets
- Face images resized and normalized

### 🔹 Model Architecture
- EfficientNet-B0 backbone
- Pretrained weights used
- Final classification layer modified for 7 classes
- Implemented using PyTorch

### 🔹 Training Process
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam
- Data augmentation applied
- Trained for multiple epochs

### 🔹 Model Performance
- Training Accuracy: 62%
- Validation Accuracy: 60%

The trained model is saved as:

model/fer_efficientnet_b0_paper.pth

and used directly in the Flask application for inference.

---

## 🧠 Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 🌐 Real-Time Web Application (`app.py`)

The Flask application performs:
- Webcam video capture
- Face detection using MTCNN
- Emotion prediction using the trained EfficientNet model
- Emotion smoothing for stable predictions
- Live emotion display on detected faces

---

## 📁 Project Structure

FER-Checking/
│
├── README.md
├── FER 2 database.ipynb
├── app.py
│
├── model/
│ └── fer_efficientnet_b0_paper.pth
│
├── templates/
│ └── index.html
│
└── static/
└── css/
└── style.css


---

## ▶️ How to Run the Application

1. Clone the repository:
```bash
git clone https://github.com/amritgupta0680/FER-Facial-Emotion-Recognition-.git
cd FER-Facial-Emotion-Recognition-

2. Run the Flask app:
```bash
python app.py

3. Open your browser:
```bash
http://127.0.0.1:5000

🧪 Test Image Example

Below is an example of emotion prediction on a test image:

⚠️ Notes

Webcam access is required

Required Python libraries must be installed

CUDA is recommended for faster inference

This project is intended for educational and research purposes

📜 License

This project is for academic and learning use.

