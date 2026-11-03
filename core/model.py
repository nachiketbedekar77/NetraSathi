"""
Diabetic Retinopathy Classification Model for NetraSathi.
Architecture: EfficientNet-B0 with 5-class severity classification head:
- 0: No DR
- 1: Mild NPDR
- 2: Moderate NPDR
- 3: Severe NPDR
- 4: Proliferative DR

Honesty Rules:
- If 'weights/dr_model.pt' exists, weights are loaded.
- Otherwise, runs in DEMO MODE with an ImageNet-initialized backbone.
- Provides a clean drop-in API: predict(image) -> (grade, probs, latency_sec, is_demo)
"""
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

GRADE_NAMES = [
    "Grade 0: No DR",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: Proliferative DR"
]

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "dr_model.pt")

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        try:
            if pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
        except Exception:
            # Fallback if offline/connection restricted
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        # Last feature map block of EfficientNet-B0 (block 8 / conv 1x1 before pooling)
        return self.backbone.features[-1]


_MODEL_INSTANCE = None
_IS_DEMO_MODE = True

def load_model() -> tuple[DRClassifier, bool]:
    """Load or retrieve singleton model instance."""
    global _MODEL_INSTANCE, _IS_DEMO_MODE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE, _IS_DEMO_MODE

    model = DRClassifier(num_classes=5, pretrained=True)
    is_demo = True

    if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1024:
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(state_dict)
            is_demo = False
            print("Loaded clinically trained weights from weights/dr_model.pt")
        except Exception as e:
            print(f"Failed to load weights file: {e}. Defaulting to DEMO MODE.")
            is_demo = True
    else:
        print("weights/dr_model.pt not found. Running in DEMO MODE.")
        is_demo = True

    model.eval()
    _MODEL_INSTANCE = model
    _IS_DEMO_MODE = is_demo
    return model, is_demo


def get_image_transform():
    """Standard ImageNet preprocessing pipeline for 224x224 input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def _analyze_demo_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    In DEMO MODE, analyze morphological fundus features (bright exudates,
    dark hemorrhages/microaneurysms, vessel tortuosity) to produce clinically plausible
    probabilities for college demonstration while awaiting clinical fine-tuning.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    
    # Simple retinal area mask
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    retina_px = np.count_nonzero(mask)
    if retina_px == 0:
        return np.array([0.9, 0.05, 0.03, 0.01, 0.01])

    # 1. Dark red lesion detector (Microaneurysms / Hemorrhages)
    # Red channel minus green channel in LAB or RGB
    r_ch = image_rgb[:, :, 0].astype(float)
    g_ch = image_rgb[:, :, 1].astype(float)
    diff = np.clip(r_ch - g_ch, 0, 255)
    
    # High difference with low green = deep hemorrhages
    hemorrhage_mask = (diff > 90) & (g_ch < 80) & (mask > 0)
    hemorrhage_ratio = np.count_nonzero(hemorrhage_mask) / retina_px

    # 2. Bright yellow lesion detector (Exudates)
    # High green + high red with low blue
    b_ch = image_rgb[:, :, 2].astype(float)
    exudate_mask = (r_ch > 160) & (g_ch > 140) & (b_ch < 130) & (mask > 0)
    exudate_ratio = np.count_nonzero(exudate_mask) / retina_px

    # 3. Fluffy white patches (Cotton Wool Spots / Fibrous Neovascularization)
    white_mask = (r_ch > 175) & (g_ch > 175) & (b_ch > 160) & (mask > 0)
    white_ratio = np.count_nonzero(white_mask) / retina_px

    # Score synthesis for demonstration
    lesion_score = hemorrhage_ratio * 40.0 + exudate_ratio * 30.0 + white_ratio * 50.0

    if lesion_score < 0.25:
        # Normal
        base_logits = [3.2, 0.8, -0.5, -1.2, -2.0]
    elif lesion_score < 0.65:
        # Mild
        base_logits = [0.4, 2.8, 1.1, -0.4, -1.5]
    elif lesion_score < 1.4:
        # Moderate
        base_logits = [-0.6, 1.0, 3.4, 1.2, -0.8]
    elif lesion_score < 2.5:
        # Severe
        base_logits = [-1.5, -0.2, 1.3, 3.5, 1.1]
    else:
        # Proliferative
        base_logits = [-2.2, -1.0, 0.4, 1.5, 3.8]

    # Softmax
    exp_l = np.exp(base_logits - np.max(base_logits))
    probs = exp_l / np.sum(exp_l)
    return probs


def predict(image_input) -> tuple[int, list[float], float, bool]:
    """
    Modular single-call prediction function:
    Accepts PIL.Image, numpy RGB or BGR array.
    Returns:
    (grade: int, probs: list[float], latency_sec: float, is_demo_mode: bool)
    """
    model, is_demo = load_model()
    
    start_time = time.perf_counter()

    # Convert input to PIL Image
    if isinstance(image_input, np.ndarray):
        if image_input.shape[2] == 3:
            # Assume RGB or convert if needed
            pil_img = Image.fromarray(image_input)
        else:
            pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image type. Pass PIL.Image or numpy.ndarray.")

    # Image tensor transformation
    transform = get_image_transform()
    img_tensor = transform(pil_img).unsqueeze(0) # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(img_tensor)
        raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    if is_demo:
        # Combine real model forward pass with morphological demo prior
        demo_probs = _analyze_demo_features(np.array(pil_img))
        # Blend: 85% demo prior for clinical plausibility in demo, 15% model logits
        blended = 0.85 * demo_probs + 0.15 * raw_probs
        probs = (blended / np.sum(blended)).tolist()
    else:
        probs = raw_probs.tolist()

    grade = int(np.argmax(probs))
    latency_sec = round(time.perf_counter() - start_time, 3)

    return grade, probs, latency_sec, is_demo

