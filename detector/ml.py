"""Model loading and inference (4 classes: COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS).

Deliberately independent of Django settings so that export_metrics.py and train.py can import it too.
PyTorch is imported lazily, so the web app starts quickly and pages that don't need the
model (home, about, history...) never pay the import cost.
"""
import threading
from pathlib import Path

# The order MUST match torchvision's ImageFolder, which sorts class folders alphabetically.
# make_imagefolder() below enforces this order whatever the folders are called on disk.
#   COVID19 -> 0, NORMAL -> 1, PNEUMONIA -> 2, TUBERCULOSIS -> 3
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = Path(__file__).resolve().parent / "ml_models" / "resnet18_chestxray_4class.pth"

# Folder-name spellings found in public datasets -> our class name (compared case-insensitively,
# ignoring spaces, hyphens and underscores). The Kaggle dataset used here spells tuberculosis
# "TURBERCULOSIS" in some versions, so both spellings are accepted.
FOLDER_ALIASES = {
    "COVID19": "COVID19", "COVID": "COVID19",
    "NORMAL": "NORMAL",
    "PNEUMONIA": "PNEUMONIA",
    "TUBERCULOSIS": "TUBERCULOSIS", "TURBERCULOSIS": "TUBERCULOSIS", "TB": "TUBERCULOSIS",
}

_lock = threading.Lock()
_model = None
_transform = None


def eval_transform():
    """The SAME transform used for validation/testing during training."""
    global _transform
    if _transform is None:
        from torchvision import transforms

        _transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _transform


def _canon(folder_name):
    key = folder_name.upper().replace("-", "").replace("_", "").replace(" ", "")
    return FOLDER_ALIASES.get(key)


def find_split_dir(root, split):
    """Return root/<split>, matching the folder name case-insensitively (Train/train/TRAIN)."""
    root = Path(root)
    for d in root.iterdir() if root.exists() else []:
        if d.is_dir() and d.name.lower() == split.lower():
            return d
    return None


def make_imagefolder(split_dir, transform):
    """ImageFolder whose classes are always CLASS_NAMES in that order.

    Raises ValueError if a class folder is missing or an unknown folder is present, so a wrong
    dataset layout fails loudly instead of silently shifting the labels.
    """
    from torchvision import datasets

    class _Folder(datasets.ImageFolder):
        def find_classes(self, directory):
            found = {}
            for d in sorted(Path(directory).iterdir()):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                name = _canon(d.name)
                if name is None:
                    raise ValueError(f"Unrecognised class folder '{d.name}' in {directory}; "
                                     f"expected folders for {CLASS_NAMES}")
                found[name] = d.name
            missing = [c for c in CLASS_NAMES if c not in found]
            if missing:
                raise ValueError(f"Missing class folder(s) {missing} in {directory}")
            # Index follows CLASS_NAMES order; the keys are the real folder names so the default
            # ImageFolder file scan works unchanged.
            real = [found[c] for c in CLASS_NAMES]
            return real, {r: i for i, r in enumerate(real)}

    ds = _Folder(str(split_dir), transform=transform)
    ds.classes = list(CLASS_NAMES)               # report canonical names, not on-disk spellings
    return ds


def build_model(pretrained=False):
    """ResNet-18 whose final layer has NUM_CLASSES outputs (shared by train.py and load_model)."""
    import torch.nn as nn
    from torchvision import models

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)        # no ImageNet download needed
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model(path=None):
    """Build ResNet-18 with a 4-way final layer and load the trained weights."""
    global _model
    path = Path(path) if path else MODEL_PATH
    if path == MODEL_PATH and _model is not None:
        return _model
    with _lock:
        if path == MODEL_PATH and _model is not None:
            return _model
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at {path}")

        import torch

        model = build_model(pretrained=False)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        if path == MODEL_PATH:
            _model = model
        return model


def predict(pil_image):
    """Return the label, its confidence and the softmax score of every class for one PIL image."""
    import torch

    model = load_model()
    img = pil_image.convert("RGB")                   # same loading step as ImageFolder
    x = eval_transform()(img).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    best = max(range(NUM_CLASSES), key=lambda i: probs[i])   # arg-max over all classes
    return {
        "label": CLASS_NAMES[best],
        "confidence": float(probs[best]),
        "scores": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
