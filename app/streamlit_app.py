import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import MultiTaskResNet

# ---------------- CONFIG ----------------
MODEL_PATH = "age_gender_model.pth"
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------

@st.cache_resource
def load_model():
    model = MultiTaskResNet(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

st.title("👁️ Age-Gaze")
st.write("AI-based Age & Gender Estimation from Faces")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5
    )

    if len(faces) == 0:
        st.warning("No faces detected.")
    else:
        for (x, y, w, h) in faces:
            face = image.crop((x, y, x + w, y + h))
            face_tensor = transform(face).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                age_pred, gender_pred = model(face_tensor)
                age = int(age_pred.item())
                gender_prob = torch.sigmoid(gender_pred).item()

            gender = "Male" if gender_prob > 0.5 else "Female"
            confidence = gender_prob if gender == "Male" else 1 - gender_prob

            label = f"{gender} ({confidence:.2f}), Age: {age-2}–{age+2}"

            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_np, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        st.image(img_np, caption="Prediction Result", use_container_width=True)
