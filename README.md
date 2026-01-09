# 🎭 Facial Emotion Recognition (FER)

A deep learning–based Facial Emotion Recognition system with a Flask web interface that detects human emotions in real time using a webcam.

---

## 📌 Project Overview

This project focuses on recognizing facial emotions from images and live video streams.  
The system is divided into **two main parts**:

1. **Model Training & Experimentation** (Jupyter Notebook)
2. **Real-Time Web Application** (Flask)

The trained deep learning model is integrated into a Flask application to perform real-time emotion detection using a webcam.

---

## 📓 Training & Dataset Details (`FER 2 database.ipynb`)

The file **`FER 2 database.ipynb`** contains the complete pipeline for training and evaluating the facial emotion recognition model. and also the **`RAFDB Dataset`** for Fine tuning the mpodel. 

### 🔹 Dataset
- Facial expression dataset with grayscale/RGB face images
- Images are categorized into **7 emotion classes**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Dataset is split into **training and validation sets**
- Face images are resized and normalized before training

### 🔹 Model Architecture
- **EfficientNet-B0** backbone
- Pretrained weights used as a base
- Final classification layer modified for **7 emotion classes**
- Implemented using **PyTorch**

### 🔹 Training Process
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam
- Data augmentation applied to improve generalization
- Model trained for multiple epochs until convergence

### 🔹 Model Performance
- Training Accuracy: 62%
- Validation Accuracy: 60%
- The trained model is saved as:
and used directly in the Flask application for inference.

*(Replace XX% with your actual accuracy if you want — optional)*

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
- Face detection using **MTCNN**
- Emotion prediction using the trained EfficientNet model
- Emotion smoothing to reduce flickering
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
git clone https://github.com/yourusername/FER-Checking.git
cd FER-Checking

---

## ▶️ How to Run the Application

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FER-Checking.git
cd FER-Checking
Run the Flask app:

python app.py


Open your browser:

http://127.0.0.1:5000

🧪 Test Image Example

Below is an example of emotion prediction on a test image:
```md
![Test Image](static/images/test.jpg)

⚠️ Notes

Webcam access is required

Required Python libraries must be installed

CUDA is recommended for faster inference

This project is intended for educational and research purposes

📜 License

This project is for academic and learning use.#

