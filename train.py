"""Train the 4-class ResNet-18 (COVID19 / NORMAL / PNEUMONIA / TUBERCULOSIS) and save it for the Django app.

This re-creates the recipe of the original 2-class notebook, with a 4-way final layer:
  * stratified 80/20 split of train/ for validation (fixed seed)
  * 3-channel grayscale, 224x224, ImageNet normalisation; augmentation on training images only
  * cross-entropy with class weights (rarer classes count more)
  * phase 1: backbone frozen, train only the new final layer (Adam, lr 1e-3)
  * phase 2: also unfreeze layer4 (Adam, lr 1e-4)
  * keep the checkpoint with the lowest validation loss

Usage (run from this folder):
    python train.py --data "C:\\path\\to\\Data"
    python export_metrics.py --data "C:\\path\\to\\Data" --log training_log.txt

--data is the folder that contains train/ and test/ (any capitalisation), each with one sub-folder per
class. The dataset's own val/ folder is ignored: validation is a stratified 20% of train/.
Needs scikit-learn in addition to requirements.txt. A GPU is used automatically if available.
"""
import argparse
import random
import sys
import time
from pathlib import Path

LOG_FILE = Path("training_log.txt")


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains train/ and test/")
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random weights instead of ImageNet weights (offline smoke tests only)")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    from detector.ml import (CLASS_NAMES, MODEL_PATH, build_model, eval_transform, find_split_dir,
                             make_imagefolder)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = find_split_dir(args.data, "train")
    if train_dir is None:
        sys.exit(f"Could not find a train/ folder inside {args.data}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        norm,
    ])
    # Two views of the same folder: augmented for training, plain for validation.
    try:
        full_train = make_imagefolder(train_dir, train_tf)
        full_val = make_imagefolder(train_dir, eval_transform())
    except ValueError as e:
        sys.exit(str(e))

    targets = np.asarray(full_train.targets)
    tr_idx, va_idx = train_test_split(np.arange(len(targets)), test_size=0.2,
                                      stratify=targets, random_state=args.seed)
    train_ds, val_ds = Subset(full_train, tr_idx), Subset(full_val, va_idx)

    counts = np.bincount(targets[tr_idx], minlength=len(CLASS_NAMES))
    weights = counts.sum() / (len(CLASS_NAMES) * counts)          # rarer class -> larger weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        log(f"device: {device}", fh)
        log("train/val images per class: " + ", ".join(
            f"{n}={c}/{int((targets[va_idx] == i).sum())}" for i, (n, c) in enumerate(zip(CLASS_NAMES, counts))), fh)
        log("class weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(CLASS_NAMES, weights)), fh)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        model = build_model(pretrained=not args.no_pretrained).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

        def run_epoch(dl, optimizer=None):
            training = optimizer is not None
            model.train(training)
            loss_sum = correct = n = 0
            with torch.set_grad_enabled(training):
                for x, y in dl:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    loss_sum += loss.item() * len(y)
                    correct += (out.argmax(1) == y).sum().item()
                    n += len(y)
            return loss_sum / n, correct / n

        best_val = float("inf")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        def phase(name, epochs, params, lr):
            nonlocal best_val
            optimizer = torch.optim.Adam(params, lr=lr)
            for ep in range(1, epochs + 1):
                t0 = time.time()
                tl, ta = run_epoch(train_dl, optimizer)
                vl, va = run_epoch(val_dl)
                # Line format is what export_metrics.py --log parses; do not change it.
                log(f"[{name}] ep {ep}/{epochs} | train loss {tl:.3f} acc {ta:.3f} | "
                    f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s", fh)
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), MODEL_PATH)

        # Phase 1: only the new final layer learns
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        phase("head", args.epochs_head, model.fc.parameters(), 1e-3)

        # Phase 2: also fine-tune the last residual block, with gentler steps
        for p in model.layer4.parameters():
            p.requires_grad = True
        phase("finetune", args.epochs_finetune,
              [p for p in model.parameters() if p.requires_grad], 1e-4)

        log(f"Saved best checkpoint (lowest val loss {best_val:.3f}) to {MODEL_PATH}", fh)
    print(f"Epoch log written to {LOG_FILE.resolve()} - pass it to export_metrics.py with --log.")


if __name__ == "__main__":
    main()
