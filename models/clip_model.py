import torch
import torch.nn as nn
import transformers
import open_clip


def build_clip_with_head(num_classes, device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    model = model.to(device)

   # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block
    for name, param in model.visual.named_parameters():
        if any(f"resblocks.{i}" in name for i in [8,9,10,11]):
            param.requires_grad = True
    # Dynamically get feature dimension
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feature_dim = model.encode_image(dummy).shape[-1]

    class CLIPHead(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, images):
            features = self.model.encode_image(images)
            return self.classifier(features)

    return CLIPHead(model).to(device), preprocess
