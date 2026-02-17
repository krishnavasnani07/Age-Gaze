# model.py
import torch
import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, shared_dim=512):
        super().__init__()
        if backbone_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            feat_dim = resnet.fc.in_features
        elif backbone_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            feat_dim = resnet.fc.in_features
        else:
            raise ValueError("Unsupported backbone")

        resnet.fc = nn.Identity()
        self.backbone = resnet

        self.shared_fc = nn.Sequential(
            nn.Linear(feat_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Age head
        self.age_fc = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Gender head
        self.gender_fc = nn.Linear(shared_dim, 1)

    def forward(self, x):
        feats = self.backbone(x)
        shared = self.shared_fc(feats)
        age_out = self.age_fc(shared).squeeze(1)    # regression (normalized)
        gender_out = self.gender_fc(shared).squeeze(1)  # logits
        return age_out, gender_out
