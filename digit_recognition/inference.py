"""Inference module for MNIST digit recognition.

Minimal deps: torch, PIL, torchvision, numpy, onnxruntime.
No dvc, hydra, or lightning imports.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _infer_model_params(state_dict):
    """Infer LeNet5 parameters from state_dict keys and shapes."""
    if not state_dict:
        return 32, 64, 128, 10, True

    conv1_w = state_dict.get("conv1.weight")
    conv1_out = conv1_w.shape[0] if conv1_w is not None else 32

    bn1_w = state_dict.get("bn1.weight")
    batch_norm_flag = bn1_w is not None

    conv2_w = state_dict.get("conv2.weight")
    conv2_out = conv2_w.shape[0] if conv2_w is not None else conv1_out * 2

    fc1_w = state_dict.get("fc1.weight")
    fc1_features = fc1_w.shape[0] if fc1_w is not None else 128

    fc2_w = state_dict.get("fc2.weight")
    fc2_features = fc2_w.shape[0] if fc2_w is not None else 10

    return conv1_out, conv2_out, fc1_features, fc2_features, batch_norm_flag


def load_model(model_path, device="cpu"):
    from digit_recognition.model import LeNet5, LeNet5WithResNet18

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    sd = checkpoint.get("state_dict", checkpoint)
    has_resnet = any("resnet" in k for k in sd)
    # Remove "model." prefix from state_dict keys if present
    if any(k.startswith("model.") for k in sd) and not any(k.startswith("resnet.") for k in sd):
        sd = {k.replace("model.", ""): v for k, v in sd.items()}

    if has_resnet:
        model = LeNet5WithResNet18(num_classes=10).to(device)
    else:
        conv1_out, conv2_out, fc1_features, fc2_features, batch_norm_flag = _infer_model_params(sd)
        model = LeNet5(
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            fc1_features=fc1_features,
            fc2_features=fc2_features,
            dropout_prob=0.5,
            batch_norm=batch_norm_flag,
        ).to(device)

    model.load_state_dict(sd)
    model.eval()
    return model


def preprocess_image(image_path, device="cpu"):
    from digit_recognition.preprocessing import normalize_grayscale, resize_image

    pil_img = Image.open(image_path).convert("L")
    arr = np.array(pil_img)
    processed = resize_image(arr, target_size=(28, 28))
    processed = normalize_grayscale(processed)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor


def predict(model, image_path, device="cpu"):
    probs = predict_with_probs(model, image_path, device=device)
    return int(torch.argmax(probs).item())


def predict_with_probs(model, image_path, device="cpu"):
    tensor = preprocess_image(image_path, device=device)
    with torch.no_grad():
        logits = model(tensor)
    return torch.softmax(logits, dim=1)
