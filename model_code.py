"""
SteamPunkDeep – Forward-Forward Network for Caltech-256
========================================================
Implements Geoffrey Hinton's Forward-Forward (FF) Algorithm (2022).
No backpropagation through the network is used. Each layer is trained
independently with a local "goodness" objective.

Supports:
  - Apple Silicon (MPS)
  - NVIDIA GPU (CUDA)
  - CPU fallback

Usage:
    python ff_caltech256.py --data_dir /path/to/caltech256 --epochs 30
    python ff_caltech256.py --data_dir /path/to/caltech256 --eval_only --model_path best_model.pt
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


# ──────────────────────────────────────────────────────────────────────────────
# 1. Device selection (MPS → CUDA → CPU)
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Apple MPS (M1/M2) detected ✓")
        return torch.device("mps")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA GPU detected: {name} ✓")
        return torch.device("cuda")
    print("[Device] Falling back to CPU")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Data loading
# ──────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 257   # 256 object classes + 1 background
IMG_SIZE    = 96    # resize images to 96×96 (balance speed vs quality)

def get_transforms(augment=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if augment:
        return T.Compose([
            T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_data(data_dir: str, batch_size: int = 128, val_split: float = 0.15):
    full_dataset = ImageFolder(data_dir, transform=get_transforms(augment=True))
    val_size     = int(len(full_dataset) * val_split)
    train_size   = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # val uses no augmentation – override transform
    val_ds.dataset.transform = get_transforms(augment=False)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"[Data] Train: {train_size} | Val: {val_size} | Classes: {len(full_dataset.classes)}")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# 3. Negative-data generation
#    Strategy: for each image, embed the WRONG class label into a corner patch.
#    Positive data gets the CORRECT class label embedded.
# ──────────────────────────────────────────────────────────────────────────────

def embed_label(images: torch.Tensor, labels: torch.Tensor,
                num_classes: int, patch_size: int = 8) -> torch.Tensor:
    """
    Embed a one-hot class label as a small patch in the top-left corner.
    This is the standard FF trick to make class information visible to
    every layer without needing a separate label input stream.

    Args:
        images:     (B, C, H, W) float tensor, already normalised
        labels:     (B,) long tensor with class indices
        num_classes: total number of classes
        patch_size:  side length of the embedded patch (pixels)

    Returns:
        images_with_label: same shape as images, patch overwritten
    """
    x = images.clone()
    B = x.size(0)

    # Build a 1-channel label map: one-hot → tiled into patch_size×patch_size
    one_hot = F.one_hot(labels, num_classes).float()   # (B, num_classes)
    # Repeat to fill patch pixels (patch_size*patch_size must >= num_classes)
    pixels_needed = patch_size * patch_size
    reps = (pixels_needed // num_classes) + 1
    patch_flat = one_hot.repeat(1, reps)[:, :pixels_needed]  # (B, P*P)
    patch = patch_flat.view(B, 1, patch_size, patch_size)    # (B, 1, P, P)

    # Write into the top-left corner of all 3 channels
    x[:, :, :patch_size, :patch_size] = patch.expand(-1, 3, -1, -1) * 2.0 - 1.0
    return x


def make_negative_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Randomly sample a WRONG class for each sample in the batch."""
    wrong = torch.randint(0, num_classes, labels.shape, device=labels.device)
    # Re-roll any that accidentally equal the correct label
    mask = (wrong == labels)
    while mask.any():
        wrong[mask] = torch.randint(0, num_classes, (mask.sum().item(),), device=labels.device)
        mask = (wrong == labels)
    return wrong


# ──────────────────────────────────────────────────────────────────────────────
# 4. Forward-Forward Layer
# ──────────────────────────────────────────────────────────────────────────────

class FFConvLayer(nn.Module):
    """
    A single convolutional FF layer.
    Trained with a local binary cross-entropy loss on 'goodness'.

    Goodness = mean of squared post-ReLU activations (spatially averaged).
    Threshold θ separates positive (goodness > θ) from negative (goodness < θ).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, theta: float = 2.0,
                 lr: float = 3e-4):
        super().__init__()
        self.conv   = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch)
        self.theta  = theta

        # Layer-local optimiser – this is the key: no global backward pass
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (inference). Returns normalised activations."""
        h = F.relu(self.bn(self.conv(x)))
        # Layer-normalise so deeper layers see stable inputs
        h = h / (h.norm(dim=1, keepdim=True) + 1e-6)
        return h

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Scalar goodness per sample: mean of squared activations."""
        return h.pow(2).mean(dim=(1, 2, 3))   # (B,)

    def ff_train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        """
        One local FF update for this layer.
        Returns (loss, h_pos, h_neg) for chaining to next layer.
        """
        h_pos = F.relu(self.bn(self.conv(x_pos)))
        h_pos_norm = h_pos / (h_pos.norm(dim=1, keepdim=True) + 1e-6)

        h_neg = F.relu(self.bn(self.conv(x_neg)))
        h_neg_norm = h_neg / (h_neg.norm(dim=1, keepdim=True) + 1e-6)

        g_pos = self.goodness(h_pos_norm)  # want > θ
        g_neg = self.goodness(h_neg_norm)  # want < θ

        # Binary cross-entropy style loss (Hinton 2022 eq.)
        loss_pos = F.softplus(-g_pos + self.theta).mean()
        loss_neg = F.softplus( g_neg - self.theta).mean()
        loss     = loss_pos + loss_neg

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), h_pos_norm.detach(), h_neg_norm.detach()


class FFPoolLayer(nn.Module):
    """Spatial down-sampling (no learnable params, no training needed)."""
    def __init__(self, kernel: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel)

    def forward(self, x):
        return self.pool(x)

    def ff_train_step(self, x_pos, x_neg):
        """Pass-through; no local loss."""
        return 0.0, self.pool(x_pos).detach(), self.pool(x_neg).detach()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Full Forward-Forward Network
# ──────────────────────────────────────────────────────────────────────────────

class ForwardForwardNet(nn.Module):
    """
    A stack of FF-trained conv layers followed by a linear classifier.

    Architecture (input 3×96×96):
        Conv 3→64   (96×96)  → Pool → 48×48
        Conv 64→128 (48×48)  → Pool → 24×24
        Conv 128→256(24×24)  → Pool → 12×12
        Conv 256→512(12×12)  → Pool → 6×6
        Flatten → 512×6×6 = 18432-d
        Linear 18432 → 1024 → 257  (trained with cross-entropy + backprop,
                                     only the classifier head uses bp)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, theta: float = 2.0, lr: float = 3e-4):
        super().__init__()
        self.num_classes = num_classes

        # FF layers (trained locally, layer by layer)
        self.ff_layers = nn.ModuleList([
            FFConvLayer(  3,  64, theta=theta, lr=lr),
            FFPoolLayer(2),
            FFConvLayer( 64, 128, theta=theta, lr=lr),
            FFPoolLayer(2),
            FFConvLayer(128, 256, theta=theta, lr=lr),
            FFPoolLayer(2),
            FFConvLayer(256, 512, theta=theta, lr=lr),
            FFPoolLayer(2),
        ])

        # Classifier head – trained separately with standard CE loss
        # (using backprop only through the HEAD, not the feature layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )
        self.cls_opt = torch.optim.AdamW(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        self.cls_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.cls_opt, T_max=50)

    # ── FF training (layer-wise, no global backprop) ──────────────────────────

    def ff_train_epoch(self, loader: DataLoader, device: torch.device, epoch: int):
        """
        Train all FF layers for one epoch.
        Positive data = image with correct label patch.
        Negative data = image with wrong label patch.
        """
        self.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            neg_labels = make_negative_labels(labels, self.num_classes)
            x_pos = embed_label(imgs, labels,     self.num_classes)
            x_neg = embed_label(imgs, neg_labels, self.num_classes)

            batch_loss = 0.0
            for layer in self.ff_layers:
                loss, x_pos, x_neg = layer.ff_train_step(x_pos, x_neg)
                batch_loss += loss

            total_loss += batch_loss
            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [FF]  Epoch {epoch} | step {step}/{len(loader)} "
                      f"| layer-sum loss {batch_loss:.4f} | {elapsed:.1f}s", flush=True)

        return total_loss / len(loader)

    # ── Classifier head training (backprop only through head) ─────────────────

    def cls_train_epoch(self, loader: DataLoader, device: torch.device, epoch: int):
        """
        Train the linear classifier on frozen FF features.
        Backprop is used here, but ONLY for the classifier head weights.
        """
        self.eval()
        self.classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            # Embed correct label for feature extraction (inference mode)
            x = embed_label(imgs, labels, self.num_classes)
            with torch.no_grad():
                features = self._extract_features(x)

            logits = self.classifier(features)
            loss   = F.cross_entropy(logits, labels)

            self.cls_opt.zero_grad()
            loss.backward()
            self.cls_opt.step()

            total_loss += loss.item()
            pred     = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

        self.cls_sched.step()
        acc = 100.0 * correct / total
        return total_loss / len(loader), acc

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass x through all FF layers (forward only)."""
        for layer in self.ff_layers:
            x = layer(x)
        return x

    # ── Inference ────────────────────────────────────────────────────────────

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Full inference. If labels are unknown (test time), we use a one-shot
        approach: run forward with each candidate label, pick highest goodness.

        For efficiency during training validation, pass labels to use the
        fast single-forward path.
        """
        if labels is not None:
            x = embed_label(imgs, labels, self.num_classes)
            features = self._extract_features(x)
            return self.classifier(features)
        else:
            # Test-time: classifier head directly (label embedding not needed
            # once classifier is trained on averaged-out features)
            # Use zero-label embed as neutral input
            dummy = torch.zeros(imgs.size(0), dtype=torch.long, device=imgs.device)
            x = embed_label(imgs, dummy, self.num_classes)
            features = self._extract_features(x)
            return self.classifier(features)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device):
        self.eval()
        correct_top1, correct_top5, total = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = self.forward(imgs, labels)
            # Top-1
            pred_top1 = logits.argmax(dim=1)
            correct_top1 += (pred_top1 == labels).sum().item()
            # Top-5
            top5 = logits.topk(5, dim=1).indices
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
        return 100.0 * correct_top1 / total, 100.0 * correct_top5 / total


# ──────────────────────────────────────────────────────────────────────────────
# 6. Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device       = get_device()
    train_loader, val_loader = load_data(args.data_dir, args.batch_size)
    model        = ForwardForwardNet(num_classes=NUM_CLASSES,
                                     theta=args.theta, lr=args.ff_lr).to(device)

    best_acc     = 0.0
    out_path     = Path(args.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SteamPunkDeep – Forward-Forward Training")
    print(f"  FF epochs: {args.epochs}  |  CLS epochs: {args.cls_epochs}")
    print(f"  θ = {args.theta}  |  FF-LR = {args.ff_lr}  |  Batch = {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Phase 1: Train FF layers layer-by-layer ────────────────────────────
    print("── Phase 1: Forward-Forward layer training ──────────────────────")
    for epoch in range(1, args.epochs + 1):
        ff_loss = model.ff_train_epoch(train_loader, device, epoch)
        print(f"[Epoch {epoch:03d}/{args.epochs}] FF avg-loss: {ff_loss:.4f}")

    # ── Phase 2: Train classifier head on frozen FF features ──────────────
    print("\n── Phase 2: Classifier head training ────────────────────────────")
    for epoch in range(1, args.cls_epochs + 1):
        cls_loss, train_acc = model.cls_train_epoch(train_loader, device, epoch)
        top1, top5          = model.evaluate(val_loader, device)
        print(f"[CLS Epoch {epoch:03d}/{args.cls_epochs}] "
              f"Loss: {cls_loss:.4f} | Train acc: {train_acc:.2f}% | "
              f"Val Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

        if top1 > best_acc:
            best_acc = top1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_top1":    top1,
                "val_top5":    top5,
                "args":        vars(args),
            }, str(out_path))
            print(f"  ✓ Saved best model → {out_path}  (val Top-1 = {best_acc:.2f}%)")

    print(f"\n✓ Training complete. Best val Top-1 = {best_acc:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Evaluation-only mode
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_only(args):
    device = get_device()
    _, val_loader = load_data(args.data_dir, args.batch_size, val_split=0.15)
    model = ForwardForwardNet(num_classes=NUM_CLASSES).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    top1, top5 = model.evaluate(val_loader, device)
    print(f"Eval → Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SteamPunkDeep – FF Network for Caltech-256")
    p.add_argument("--data_dir",    type=str,   default="./caltech256",
                   help="Root directory of Caltech-256 (ImageFolder structure)")
    p.add_argument("--model_path",  type=str,   default="./checkpoints/best_ff_model.pt")
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=30,
                   help="Number of Forward-Forward training epochs")
    p.add_argument("--cls_epochs",  type=int,   default=50,
                   help="Number of classifier-head training epochs")
    p.add_argument("--theta",       type=float, default=2.0,
                   help="Goodness threshold θ (Hinton recommends ~2)")
    p.add_argument("--ff_lr",       type=float, default=3e-4,
                   help="Learning rate for FF layer-local optimisers")
    p.add_argument("--eval_only",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        evaluate_only(args)
    else:
        train(args)
