import cv2
import torch
from PIL import Image
from torchvision import transforms
from model import MultiTaskResNet

# ---------------- SETTINGS ----------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Transform
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# Load model
def load_model(path="age_gender_model.pth"):
    model = MultiTaskResNet(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Predict
def predict(model, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        # fallback: resize entire image
        face = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        x, y, w, h = 0, 0, IMG_SIZE, IMG_SIZE
    else:
        x, y, w, h = faces[0]
        face = img_bgr[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    # Convert to tensor
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = tfm(pil_img).unsqueeze(0).to(DEVICE, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        raw_age, gender_logits = model(input_tensor)
        age = raw_age.item() * 100  # denormalize
        gender_prob = torch.sigmoid(gender_logits).item()
        gender = "Male" if gender_prob < 0.5 else "Female"

    # Draw face box & label
    output_img = img_bgr.copy()
    cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(output_img, f"{gender} ({gender_prob:.2f}) Age:{age:.1f}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return age, gender, gender_prob, output_img
