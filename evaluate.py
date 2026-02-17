# evaluate.py
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from dataset_loader import UTKFaceDataset
    from model import MultiTaskResNet
    from torchvision import transforms
    from tqdm import tqdm
    import numpy as np
    import torch.multiprocessing as mp

    mp.freeze_support()

    DATA_DIR = "C:\\Users\\Nipun\\Desktop\\resusme project\\dataset\\UTKFace"
    MODEL_PATH = "age_gender_model.pth"
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 128
    MAX_AGE = 116.0

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    dataset = UTKFaceDataset(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    total = 0
    correct_gender = 0
    mae_sum = 0.0

    with torch.no_grad():
        for imgs, ages, genders in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            ages = ages.to(DEVICE)     # normalized
            genders = genders.to(DEVICE)

            pred_age, pred_gender = model(imgs)
            pred_age_years = pred_age * MAX_AGE
            ages_years = ages * MAX_AGE

            pred_gender_label = (torch.sigmoid(pred_gender) >= 0.5).float()
            correct_gender += (pred_gender_label == genders).sum().item()

            mae_sum += torch.sum(torch.abs(pred_age_years - ages_years)).item()
            total += imgs.size(0)

    print(f"Gender Accuracy: {100 * correct_gender / total:.2f}%")
    print(f"Age MAE: {mae_sum / total:.2f} years")
