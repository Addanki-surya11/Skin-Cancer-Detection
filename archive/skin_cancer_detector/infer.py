import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from xgboost import XGBClassifier


def compute_skin_ratio(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    rule1 = (r > 95) & (g > 40) & (b > 20) & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    rule2 = (r > 220) & (g > 210) & (b > 170) & (np.abs(r - g) <= 15) & (r > b) & (g > b)

    return float((rule1 | rule2).mean())


class ResNetEmbedder(nn.Module):
    def __init__(self, trained_resnet: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(trained_resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class SkinCancerDetector:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir).resolve()
        with open(self.model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        payload = torch.load(self.model_dir / "cnn_model.pt", map_location=self.device)
        self.cnn = self._create_cnn(payload["num_classes"])
        self.cnn.load_state_dict(payload["state_dict"])
        self.cnn.to(self.device)
        self.cnn.eval()

        self.embedder = ResNetEmbedder(self.cnn)
        self.embedder.to(self.device)
        self.embedder.eval()

        self.xgb = XGBClassifier()
        self.xgb.load_model(str(self.model_dir / "xgb_model.json"))
        # XGBoost sklearn wrapper may lose class metadata on load.
        # Some versions allow setting these attributes, some expose them as read-only.
        try:
            self.xgb.n_classes_ = len(self.class_names)
            self.xgb.classes_ = np.arange(len(self.class_names))
        except Exception:
            pass

        self.ood = joblib.load(self.model_dir / "ood_isolation_forest.joblib")

        self.eval_tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _create_cnn(self, num_classes: int) -> nn.Module:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def _load_image(self, image_source: Any) -> Image.Image:
        if isinstance(image_source, (str, Path)):
            return Image.open(image_source).convert("RGB")
        if isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source)).convert("RGB")
        raise TypeError("image_source must be a file path or raw bytes")

    def predict(self, image_source: Any) -> Dict[str, Any]:
        image = self._load_image(image_source)

        skin_ratio = compute_skin_ratio(image)
        skin_thr = float(self.meta["rejection"]["skin_ratio_threshold"])
        if skin_ratio < skin_thr:
            return {
                "accepted": False,
                "reason": "non_skin_rejected_by_skin_ratio",
                "skin_ratio": skin_ratio,
                "threshold": skin_thr,
            }

        x = self.eval_tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedder(x).cpu().numpy()[0]

        probs = self.xgb.predict_proba(emb.reshape(1, -1))[0]
        if len(probs) != len(self.class_names):
            # Fallback: use booster-level prediction to get full multiclass softprob output.
            booster_probs = self.xgb.get_booster().inplace_predict(emb.reshape(1, -1))
            probs = np.asarray(booster_probs)[0]
        if len(probs) != len(self.class_names):
            raise RuntimeError(
                f"Model class mismatch: got {len(probs)} probabilities for {len(self.class_names)} classes. "
                "Re-train and re-save model artifacts."
            )
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        ood_score = float(self.ood.decision_function([emb])[0])
        ood_thr = float(self.meta["rejection"]["ood_threshold"])
        ood_bypass = float(self.meta["rejection"].get("ood_confidence_bypass", 0.75))
        if ood_score < ood_thr and top_prob < ood_bypass:
            return {
                "accepted": False,
                "reason": "non_skin_or_ood_rejected",
                "ood_score": ood_score,
                "threshold": ood_thr,
                "confidence": top_prob,
                "ood_confidence_bypass": ood_bypass,
            }

        confidence_thr = float(self.meta["rejection"]["confidence_threshold"])

        if top_prob < confidence_thr:
            return {
                "accepted": False,
                "reason": "low_confidence_rejected",
                "confidence": top_prob,
                "threshold": confidence_thr,
            }

        return {
            "accepted": True,
            "predicted_class": self.class_names[top_idx],
            "confidence": top_prob,
            "all_probabilities": {c: float(p) for c, p in zip(self.class_names, probs.tolist())},
            "skin_ratio": skin_ratio,
            "ood_score": ood_score,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for skin cancer detector")
    parser.add_argument("--model-dir", type=str, default="./model_artifacts")
    parser.add_argument("--image", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = SkinCancerDetector(args.model_dir)
    result = detector.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
