import streamlit as st
import cv2
import numpy as np
from inference import load_model, predict

st.set_page_config(page_title="Age & Gender Predictor", layout="centered")
st.title("🧠 Age & Gender Prediction")

# Sidebar
model_path = st.sidebar.text_input("Model Path", value="age_gender_model.pth")
model = load_model(model_path)

# Image source
option = st.radio("Select Image Source:", ("Upload Image", "Camera"))
image = None

if option == "Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        bytes_data = camera_input.read()
        file_bytes = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Prediction
if image is not None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image", use_column_width=True)
    with st.spinner("Predicting..."):
        age, gender, prob, face = predict(model, image)

    st.success(f"Predicted Age: {age:.1f} years")
    st.info(f"Gender: {gender} ({prob:.2f} confidence)")
    st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Detected Face")
else:
    st.warning("Please upload or capture an image.")
