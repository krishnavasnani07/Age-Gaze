# train.py
if __name__ == "__main__":
    import os
    import torch
    import yaml 
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.dataset_loader import UTKFaceDataset
    from src.model import MultiTaskResNet
    from tqdm import tqdm
    import torch.multiprocessing as mp

    mp.freeze_support()  # Windows safe

    # -------- CONFIG LOADING --------
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "config.yaml"
        )

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Training params
    EPOCHS = cfg["training"]["epochs"]
    BATCH_SIZE = cfg["training"]["batch_size"]
    LR = cfg["training"]["learning_rate"]
    IMG_SIZE = cfg["training"]["img_size"]
    NUM_WORKERS = cfg["training"]["num_workers"]

    # Model params
    FREEZE_BACKBONE = cfg["model"]["freeze_backbone"]

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, cfg["paths"]["dataset_dir"])
    MODEL_PATH = cfg["paths"]["model_output"]

    # Device
    DEVICE = "cuda" if (
        cfg["device"]["use_cuda"] and torch.cuda.is_available()
    ) else "cpu"

    print(f"Using device: {DEVICE}")


    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    dataset = UTKFaceDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = MultiTaskResNet(pretrained=True).to(DEVICE)
    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    age_criterion = nn.SmoothL1Loss()   # robust regression
    gender_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for imgs, ages, genders in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE, dtype=torch.float32)
            ages = ages.to(DEVICE, dtype=torch.float32)
            genders = genders.to(DEVICE, dtype=torch.float32)

            optimizer.zero_grad()
            pred_age, pred_gender = model(imgs)
            loss_age = age_criterion(pred_age, ages)
            loss_gender = gender_criterion(pred_gender, genders)
            loss = loss_age + loss_gender

            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")
