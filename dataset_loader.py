# dataset_loader.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

MAX_AGE = 116.0  # UTKFace max age

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        parts = img_name.split("_")
        try:
            age = float(parts[0])
            gender = int(parts[1])
        except Exception:
            # fallback if filename unexpected
            age, gender = 0.0, 0

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        age = torch.tensor(age / MAX_AGE, dtype=torch.float32)   # normalized to [0,1]
        gender = torch.tensor(gender, dtype=torch.float32)       # 0/1 float

        return image, age, gender
