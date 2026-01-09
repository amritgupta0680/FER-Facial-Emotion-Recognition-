import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs

import cv2
import torch
import torch.nn.functional as F
from flask import Flask, render_template, Response
from torchvision import transforms
from mtcnn import MTCNN
from collections import deque, Counter
import timm

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Emotion Labels
# -------------------------------
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "model/fer_efficientnet_b0_paper.pth"

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(EMOTIONS)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Face Detector
# -------------------------------
detector = MTCNN()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Emotion Smoothing
# -------------------------------
emotion_queue = deque(maxlen=7)

# -------------------------------
# Video Generator
# -------------------------------
def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for face in faces:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_tensor = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)
                emotion_idx = torch.argmax(probs).item()
                emotion = EMOTIONS[emotion_idx]

            emotion_queue.append(emotion)
            final_emotion = Counter(emotion_queue).most_common(1)[0][0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                final_emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

    cap.release()

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
