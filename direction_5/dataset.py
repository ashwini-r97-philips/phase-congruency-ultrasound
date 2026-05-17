"""TN3K dataset loader from HuggingFace or local disk."""

import os
import re
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import yaml


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── HuggingFace loader ────────────────────────────────────────────────────────

def _numeric_key(entry):
    """Extract numeric index from HF entry filename for stable sorting."""
    fname = None
    # Try common filename fields
    for key in ("file_name", "image_file_name", "mask_file_name"):
        if key in entry:
            fname = entry[key]
            break
    if fname is None:
        return 0
    m = re.search(r"(\d+)", str(fname))
    return int(m.group(1)) if m else 0


def load_tn3k_from_hf(cfg):
    """Load TN3K from HuggingFace. Returns (image_pil_list, mask_pil_list).

    haifan-gong/TN3K stores images and masks in an imagefolder layout.
    The dataset schema is inspected at runtime to handle both possible
    structures:
      - 'image' + 'mask' columns (segmentation format)
      - 'image' + 'label' ClassLabel (image/mask interleaved by label value)
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"]["hf_path"], trust_remote_code=True)

    # Pick whichever split exists
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    split = ds[split_name]

    col_names = split.column_names
    print(f"[dataset] HF split='{split_name}', columns={col_names}, rows={len(split)}")

    # Case 1: explicit 'mask' column
    if "mask" in col_names:
        images = [row["image"].convert("L") for row in split]
        masks = [row["mask"].convert("L") for row in split]
        print(f"[dataset] Found 'mask' column — {len(images)} pairs")
        return images, masks

    # Case 2: 'label' ClassLabel separates images (0) from masks (1)
    if "label" in col_names:
        img_rows = [row for row in split if row["label"] == 0]
        msk_rows = [row for row in split if row["label"] == 1]
        img_rows.sort(key=_numeric_key)
        msk_rows.sort(key=_numeric_key)
        if len(img_rows) == len(msk_rows) and len(img_rows) > 0:
            images = [row["image"].convert("L") for row in img_rows]
            masks = [row["image"].convert("L") for row in msk_rows]
            print(f"[dataset] Interleaved layout — {len(images)} pairs via label split")
            return images, masks

    # Case 3: alternating rows (no label field), assume even=image odd=mask
    if len(split) % 2 == 0:
        images = [split[i]["image"].convert("L") for i in range(0, len(split), 2)]
        masks = [split[i]["image"].convert("L") for i in range(1, len(split), 2)]
        print(f"[dataset] Alternating layout assumed — {len(images)} pairs")
        return images, masks

    raise ValueError(
        f"Cannot parse TN3K HF dataset. Columns: {col_names}. "
        "Set dataset.local_root in config.yaml to use a local copy."
    )


# ── Local disk loader ─────────────────────────────────────────────────────────

def _find_file(directory, stem):
    """Find a file in `directory` whose stem matches `stem`, any extension."""
    for p in directory.iterdir():
        if p.stem == str(stem):
            return p
    return None


def load_tn3k_from_local(cfg):
    """Load TN3K from the official local layout:

      local_root/
        trainval-image/   <id>.png   (ids from fold JSON "train" + "val" lists)
        trainval-mask/    <id>.png
        test-image/       <id>.png   (fixed held-out test set)
        test-mask/        <id>.png
        tn3k_trainval_fold{N}.json   (keys: "train", "val", "test")

    Returns dict:
      {
        "train": (img_path_list, mask_path_list),
        "val":   (img_path_list, mask_path_list),
        "test":  (img_path_list, mask_path_list),
      }
    """
    dcfg = cfg["dataset"]
    root = Path(dcfg["local_root"])
    fold = dcfg.get("fold", 0)

    fold_json = root / f"tn3k_trainval_fold{fold}.json"
    if not fold_json.exists():
        # Try alternate naming conventions
        candidates = list(root.glob("*fold*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No fold JSON found in {root}. Expected tn3k_trainval_fold{{N}}.json"
            )
        fold_json = sorted(candidates)[fold] if fold < len(candidates) else candidates[0]
        print(f"[dataset] Using fold file: {fold_json.name}")

    with open(fold_json) as f:
        splits = json.load(f)

    trainval_img_dir = root / "trainval-image"
    trainval_msk_dir = root / "trainval-mask"
    test_img_dir = root / "test-image"
    test_msk_dir = root / "test-mask"

    def _build_items(ids, img_dir, msk_dir, split_name):
        pairs = [(i, _find_file(img_dir, i), _find_file(msk_dir, i)) for i in ids]
        missing = [i for i, ip, mp in pairs if ip is None or mp is None]
        found = [(ip, mp) for _, ip, mp in pairs if ip is not None and mp is not None]
        if missing:
            preview = str(missing[:5]) + ("..." if len(missing) > 5 else "")
            print(f"[dataset] WARNING: {len(missing)} missing files in '{split_name}': {preview}")
        print(f"[dataset] Local '{split_name}': {len(found)} pairs (fold {fold})")
        img_paths = [ip for ip, _ in found]
        msk_paths = [mp for _, mp in found]
        return img_paths, msk_paths

    result = {}
    result["train"] = _build_items(splits["train"], trainval_img_dir, trainval_msk_dir, "train")
    result["val"] = _build_items(splits["val"], trainval_img_dir, trainval_msk_dir, "val")

    # Test set: use dedicated test-image/test-mask dirs; ignore the empty "test" key in JSON
    test_imgs = sorted(test_img_dir.glob("*"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    test_msks = sorted(test_msk_dir.glob("*"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    assert len(test_imgs) == len(test_msks), (
        f"Test image/mask count mismatch: {len(test_imgs)} vs {len(test_msks)}"
    )
    print(f"[dataset] Local 'test': {len(test_imgs)} pairs")
    result["test"] = (test_imgs, test_msks)

    return result


# ── Dataset class ─────────────────────────────────────────────────────────────

class TN3KDataset(Dataset):
    def __init__(self, items, image_size, augment=False):
        """
        items: list of (pil_image_or_path, pil_mask_or_path, image_id)
        image_size: (H, W)
        """
        self.items = items
        self.image_size = tuple(image_size)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_src, msk_src, image_id = self.items[idx]

        # Load
        if isinstance(img_src, (str, Path)):
            image = Image.open(img_src).convert("L")
            mask = Image.open(msk_src).convert("L")
        else:
            image = img_src.convert("L")
            mask = msk_src.convert("L")

        orig_size = image.size  # (W, H)

        # Resize to training size
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_t = torch.from_numpy(image).unsqueeze(0)   # [1,H,W]
        mask_t = torch.from_numpy(mask).unsqueeze(0)     # [1,H,W]

        return {
            "image": image_t,
            "mask": mask_t,
            "id": str(image_id),
            "orig_size": orig_size,  # (W, H)
        }

    def _augment(self, image, mask):
        image_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        # Random horizontal flip
        if random.random() > 0.5:
            image_t = TF.hflip(image_t)
            mask_t = TF.hflip(mask_t)

        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        image_t = TF.rotate(image_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask_t = TF.rotate(mask_t, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Brightness jitter (image only)
        factor = random.uniform(0.8, 1.2)
        image_t = torch.clamp(image_t * factor, 0.0, 1.0)

        return image_t.squeeze(0).numpy(), mask_t.squeeze(0).numpy()


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloaders(cfg_path):
    """Build train/val/test DataLoaders from config.

    Returns dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    cfg = _load_cfg(cfg_path)
    dcfg = cfg["dataset"]
    image_size = dcfg["image_size"]
    seed = dcfg["random_seed"]
    rng = random.Random(seed)

    if dcfg.get("local_root") and dcfg.get("use_official_split"):
        splits = load_tn3k_from_local(cfg)
        train_items = list(zip(splits["train"][0], splits["train"][1],
                               [p.stem for p in splits["train"][0]]))
        val_items = list(zip(splits["val"][0], splits["val"][1],
                             [p.stem for p in splits["val"][0]]))
        test_items = list(zip(splits["test"][0], splits["test"][1],
                              [p.stem for p in splits["test"][0]]))
    else:
        images, masks = load_tn3k_from_hf(cfg)
        ids = [str(i).zfill(4) for i in range(len(images))]
        all_items = list(zip(images, masks, ids))
        rng.shuffle(all_items)

        fracs = dcfg["train_val_test_split"]
        n = len(all_items)
        n_train = int(n * fracs[0])
        n_val = int(n * fracs[1])
        train_items = all_items[:n_train]
        val_items = all_items[n_train:n_train + n_val]
        test_items = all_items[n_train + n_val:]

    print(f"[dataset] train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    num_workers = dcfg.get("num_workers", 4)

    loaders = {
        "train": DataLoader(
            TN3KDataset(train_items, image_size, augment=True),
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            TN3KDataset(val_items, image_size, augment=False),
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            TN3KDataset(test_items, image_size, augment=False),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        ),
    }
    return loaders


def get_test_items(cfg_path):
    """Return the raw test item list (image_src, mask_src, id) without DataLoader."""
    cfg = _load_cfg(cfg_path)
    dcfg = cfg["dataset"]
    seed = dcfg["random_seed"]
    rng = random.Random(seed)

    if dcfg.get("local_root") and dcfg.get("use_official_split"):
        splits = load_tn3k_from_local(cfg)
        test_imgs, test_msks = splits["test"]
        return list(zip(test_imgs, test_msks, [p.stem for p in test_imgs]))

    else:
        images, masks = load_tn3k_from_hf(cfg)
        ids = [str(i).zfill(4) for i in range(len(images))]
        all_items = list(zip(images, masks, ids))
        rng.shuffle(all_items)
        fracs = dcfg["train_val_test_split"]
        n = len(all_items)
        n_train = int(n * fracs[0])
        n_val = int(n * fracs[1])
        return all_items[n_train + n_val:]


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    loaders = make_dataloaders(cfg_path)
    batch = next(iter(loaders["train"]))
    print("image shape:", batch["image"].shape)
    print("mask shape:", batch["mask"].shape)
    print("ids:", batch["id"][:3])
    print("mask unique values:", batch["mask"].unique())
