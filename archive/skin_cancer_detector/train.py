import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from xgboost import XGBClassifier


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        y = self.label_to_idx[row["label"]]
        return image, y


class ImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def collect_data(data_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for class_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for image_path in class_dir.glob("*.jpg"):
            rows.append({"image_path": str(image_path), "label": label})

    if not rows:
        raise ValueError(f"No JPG images found under {data_root}")

    df = pd.DataFrame(rows)
    return df


def make_splits(df: pd.DataFrame) -> SplitData:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


def make_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[nn.Module, float]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val_acc = 0.0
    no_improve = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_acc = evaluate_cnn_accuracy(model, val_loader, device)
        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 3:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Model did not train correctly.")

    model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate_cnn_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    return accuracy_score(all_true, all_preds)


class ResNetEmbedder(nn.Module):
    def __init__(self, trained_resnet: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(trained_resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


def extract_embeddings(
    embedder: nn.Module,
    df: pd.DataFrame,
    transform,
    label_to_idx: Dict[str, int],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = SkinDataset(df=df, label_to_idx=label_to_idx, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embedder.to(device)
    embedder.eval()

    embs: List[np.ndarray] = []
    labels: List[int] = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extract embeddings", leave=False):
            x = x.to(device)
            e = embedder(x).cpu().numpy()
            embs.append(e)
            labels.extend(y.numpy().tolist())

    return np.concatenate(embs, axis=0), np.array(labels, dtype=np.int64)


def compute_skin_ratio(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    rule1 = (r > 95) & (g > 40) & (b > 20) & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    rule2 = (r > 220) & (g > 210) & (b > 170) & (np.abs(r - g) <= 15) & (r > b) & (g > b)

    skin_mask = rule1 | rule2
    return float(skin_mask.mean())


def estimate_skin_threshold(train_df: pd.DataFrame) -> float:
    ratios: List[float] = []
    for p in tqdm(train_df["image_path"].tolist(), desc="Estimate skin threshold", leave=False):
        try:
            image = Image.open(p).convert("RGB")
            ratios.append(compute_skin_ratio(image))
        except Exception:
            continue

    if not ratios:
        return 0.03

    thr = float(np.percentile(ratios, 2))
    return max(0.03, min(0.60, thr))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train skin cancer detector: CNN feature extractor + XGBoost classifier")
    parser.add_argument("--data-dir", type=str, default="../sorted_by_dx", help="Path to sorted class folders")
    parser.add_argument("--output-dir", type=str, default="./model_artifacts", help="Where trained artifacts are saved")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    df = collect_data(data_dir)
    print(df["label"].value_counts().sort_index())

    splits = make_splits(df)
    class_names = sorted(df["label"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(class_names)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    train_tf, eval_tf = make_transforms()

    train_ds = SkinDataset(splits.train_df, label_to_idx, transform=train_tf)
    val_ds = SkinDataset(splits.val_df, label_to_idx, transform=eval_tf)
    test_ds = SkinDataset(splits.test_df, label_to_idx, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_counts = splits.train_df["label"].value_counts().reindex(class_names).values
    class_weights_np = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn_model = create_model(num_classes=len(class_names))
    cnn_model, best_val_acc = train_cnn(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    cnn_test_acc = evaluate_cnn_accuracy(cnn_model, test_loader, device)
    print(f"Best CNN val accuracy: {best_val_acc:.4f}")
    print(f"CNN test accuracy: {cnn_test_acc:.4f}")

    embedder = ResNetEmbedder(cnn_model)
    x_train, y_train = extract_embeddings(embedder, splits.train_df, eval_tf, label_to_idx, device, args.batch_size)
    x_val, y_val = extract_embeddings(embedder, splits.val_df, eval_tf, label_to_idx, device, args.batch_size)
    x_test, y_test = extract_embeddings(embedder, splits.test_df, eval_tf, label_to_idx, device, args.batch_size)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    xgb_pred = xgb.predict(x_test)
    xgb_probs_test = xgb.predict_proba(x_test)
    xgb_test_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost(test) accuracy: {xgb_test_acc:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=class_names, digits=4))

    xgb_probs_val = xgb.predict_proba(x_val)
    correct_val_probs = xgb_probs_val[np.arange(len(y_val)), y_val]
    confidence_threshold = float(max(0.45, np.percentile(correct_val_probs, 10)))

    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=SEED)
    iso.fit(x_train)
    val_scores = iso.decision_function(x_val)
    ood_threshold = float(np.percentile(val_scores, 1))

    skin_ratio_threshold = estimate_skin_threshold(splits.train_df)

    model_payload = {
        "state_dict": cnn_model.state_dict(),
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    torch.save(model_payload, out_dir / "cnn_model.pt")
    xgb.save_model(str(out_dir / "xgb_model.json"))
    joblib.dump(iso, out_dir / "ood_isolation_forest.joblib")

    metadata = {
        "class_names": class_names,
        "idx_to_label": idx_to_label,
        "cnn_val_accuracy": best_val_acc,
        "cnn_test_accuracy": cnn_test_acc,
        "xgb_test_accuracy": float(xgb_test_acc),
        "target_accuracy": float(args.target_accuracy),
        "meets_target": bool(xgb_test_acc >= args.target_accuracy),
        "rejection": {
            "confidence_threshold": confidence_threshold,
            "ood_threshold": ood_threshold,
            "ood_confidence_bypass": 0.75,
            "skin_ratio_threshold": skin_ratio_threshold,
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to: {out_dir}")
    if xgb_test_acc < args.target_accuracy:
        print(
            "Target accuracy not reached. Try: more data, class balancing, longer training, or stronger CNN backbone."
        )


if __name__ == "__main__":
    main()
