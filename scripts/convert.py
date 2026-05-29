"""Convert a trained PyTorch checkpoint to ONNX + verify against PyTorch."""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def export_onnx(model_path: str, onnx_path: str, input_shape_str: str = "1,1,28,28") -> None:
    """Load PyTorch checkpoint -> export to ONNX -> simplify."""
    from digit_recognition.inference import _infer_model_params
    from digit_recognition.model import LeNet5, LeNet5WithResNet18

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    sd = checkpoint.get("state_dict", checkpoint)

    # Remove "model." prefix from state_dict keys if present
    if any(k.startswith("model.") for k in sd) and not any(k.startswith("resnet.") for k in sd):
        sd = {k.replace("model.", ""): v for k, v in sd.items()}

    has_resnet = any("resnet" in k for k in sd)

    if has_resnet:
        model = LeNet5WithResNet18(num_classes=10)
    else:
        conv1_out, conv2_out, fc1_features, fc2_features, batch_norm_flag = _infer_model_params(sd)
        model = LeNet5(
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            fc1_features=fc1_features,
            fc2_features=fc2_features,
            dropout_prob=0.5,
            batch_norm=batch_norm_flag,
        )
    model.load_state_dict(sd)
    model.eval()

    input_shape = tuple(int(x) for x in input_shape_str.split(","))
    onnx_path = Path(onnx_path)
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=14,
    )

    onnx_model = onnx.load(str(onnx_path))
    simplified, ok = onnxsim.simplify(onnx_model)
    assert ok, "Simplified ONNX model failed verification"
    onnx.save(simplified, str(onnx_path))
    print(f"ONNX exported to {onnx_path} ({onnx_path.stat().st_size} bytes)")


def test_onnx_consistency(
    model_path: str,
    onnx_path: str,
    test_image: str | None = None,
) -> None:
    """Compare PyTorch vs ONNX predictions on a random MNIST sample."""
    from digit_recognition.inference import preprocess_image

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    sd = checkpoint.get("state_dict", checkpoint)

    # Remove "model." prefix from state_dict keys if present
    if any(k.startswith("model.") for k in sd) and not any(k.startswith("resnet.") for k in sd):
        sd = {k.replace("model.", ""): v for k, v in sd.items()}

    has_resnet = any("resnet" in k for k in sd)
    if has_resnet:
        from digit_recognition.model import LeNet5WithResNet18

        model = LeNet5WithResNet18(num_classes=10)
    else:
        from digit_recognition.inference import _infer_model_params
        from digit_recognition.model import LeNet5

        conv1_out, conv2_out, fc1_features, fc2_features, batch_norm_flag = _infer_model_params(sd)
        model = LeNet5(
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            fc1_features=fc1_features,
            fc2_features=fc2_features,
            dropout_prob=0.5,
            batch_norm=batch_norm_flag,
        )
    model.load_state_dict(sd)
    model.eval()

    # Prepare input
    if test_image and Path(test_image).exists():
        input_tensor = preprocess_image(test_image, device="cpu")
    else:
        dataset = MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        sample_image, _ = dataset[42]
        # sample_image is (1, 28, 28) -> need (1, 1, 28, 28)
        input_tensor = sample_image.unsqueeze(0)

    with torch.no_grad():
        pt_probs = torch.softmax(model(input_tensor), dim=1)
        pt_arr = pt_probs.detach().numpy().flatten()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    np_input = input_tensor.cpu().numpy().astype("float32")
    ort_logits = session.run(None, {input_name: np_input})[0]
    ort_logits_flat = ort_logits[0]

    e_x = np.exp(ort_logits_flat - np.max(ort_logits_flat))
    ort_probs = e_x / e_x.sum()
    ort_class = int(np.argmax(ort_probs))

    max_diff = float(np.max(np.abs(pt_arr - ort_probs)))
    pt_class = int(np.argmax(pt_arr))

    print(f"PyTorch class: {pt_class}  ONNX class: {ort_class}")
    print(f"Max prob diff: {max_diff:.6f}")
    assert pt_class == ort_class, f"Class mismatch: PT={pt_class} != ONNX={ort_class}"
    print("OK - ONNX inference matches PyTorch")


if __name__ == "__main__":
    fire.Fire()
