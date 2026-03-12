#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert PNG images (with or without masks) to NPZ files for Swin-Unet.

Key behavior:
- No dataset splitting.
- Every run converts all discovered images to NPZ.
- The NPZ files generated in this run are written to one target split txt
  (train.txt or test.txt) under lists/<dataset_name>/.
- If no mask PNG exists, use --no_masks to create all-zero labels.
- Always keeps RGB 3 channels and supports configurable normalization.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Convert PNG image/mask pairs to NPZ for Swin-Unet")

    parser.add_argument("--images_dir", type=str, required=True, help="Path to image PNG directory")
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="",
        help="Path to mask PNG directory. Optional when --no_masks is enabled",
    )
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory")
    parser.add_argument("--dataset_name", type=str, default="MyDataset", help="List folder name under out_root/lists/")
    parser.add_argument("--img_suffix", type=str, default=".png", help="Image suffix")
    parser.add_argument("--mask_suffix", type=str, default=".png", help="Mask suffix")

    parser.add_argument(
        "--target_split",
        type=str,
        choices=["train", "test", "val"],
        required=True,
        help="Write all NPZ files generated in this run to this split txt",
    )
    parser.add_argument(
        "--write_mode",
        type=str,
        choices=["overwrite", "append"],
        default="overwrite",
        help="How to write split txt for this run",
    )

    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Use all-zero labels and do not read mask PNG files",
    )

    parser.add_argument(
        "--norm",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="Image normalization type",
    )
    parser.add_argument(
        "--zscore_scope",
        type=str,
        default="per_image",
        choices=["per_image", "dataset"],
        help="Scope of z-score statistics (only used when --norm zscore)",
    )

    parser.add_argument(
        "--label_map",
        type=str,
        default="",
        help='Optional label remap, e.g. "0:0,128:1,255:2". '
             "If empty and --keep_mask_values is not set, auto map unique values to 0..K-1.",
    )
    parser.add_argument(
        "--keep_mask_values",
        action="store_true",
        help="Keep original mask values (use only if already 0..K-1).",
    )

    parser.add_argument(
        "--all_txt",
        action="store_true",
        help="Also write all generated stems to all.txt (overwrite).",
    )

    return parser.parse_args()


def collect_samples(images_dir: Path, masks_dir: Path, img_suffix: str, mask_suffix: str, no_masks: bool):
    image_paths = sorted(images_dir.glob(f"*{img_suffix}"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in {images_dir} with suffix {img_suffix}")

    samples = []
    if no_masks:
        for img_p in image_paths:
            samples.append((img_p, None, img_p.stem))
        return samples

    missing = []
    for img_p in image_paths:
        stem = img_p.stem
        mask_p = masks_dir / f"{stem}{mask_suffix}"
        if mask_p.exists():
            samples.append((img_p, mask_p, stem))
        else:
            missing.append(stem)

    if missing:
        print(f"[WARN] {len(missing)} masks missing, skipped. Examples: {missing[:10]}")
    if len(samples) == 0:
        raise RuntimeError("No valid image-mask pairs found.")
    return samples


def parse_label_map(label_map_str: str):
    if not label_map_str:
        return None
    mapping = {}
    for item in label_map_str.split(","):
        item = item.strip()
        if not item:
            continue
        src, dst = item.split(":")
        mapping[int(src)] = int(dst)
    return mapping


def remap_mask(mask: np.ndarray, mapping: dict):
    out = np.zeros_like(mask, dtype=np.int16)
    for src, dst in mapping.items():
        out[mask == src] = dst
    return out


def load_image(img_path: Path):
    return np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)


def minmax_norm(img: np.ndarray):
    return img / 255.0


def zscore_per_image(img: np.ndarray):
    if img.ndim == 2:
        mean = float(img.mean())
        std = float(img.std())
        std = std if std > 1e-8 else 1.0
        return (img - mean) / std

    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        channel = img[..., c]
        mean = float(channel.mean())
        std = float(channel.std())
        std = std if std > 1e-8 else 1.0
        out[..., c] = (channel - mean) / std
    return out


def compute_dataset_zscore_stats(samples):
    # use_mask_for_norm = [false, false, false], i.e. full image statistics only.
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for img_p, _, _ in samples:
        img = load_image(img_p)  # H,W,3 float32
        flat = img.reshape(-1, 3).astype(np.float64)
        sum_c += flat.sum(axis=0)
        sumsq_c += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]
    mean = sum_c / max(n_pixels, 1)
    var = (sumsq_c / max(n_pixels, 1)) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_with_dataset_stats(img: np.ndarray, mean, std):
    if img.ndim == 2:
        return (img - mean) / (std if float(std) > 1e-8 else 1.0)
    return (img - mean.reshape(1, 1, -1)) / np.where(std.reshape(1, 1, -1) > 1e-8, std.reshape(1, 1, -1), 1.0)


def normalize_image(img: np.ndarray, norm: str, zscore_scope: str, dataset_stats=None):
    if norm == "minmax":
        return minmax_norm(img).astype(np.float32)

    if zscore_scope == "per_image":
        return zscore_per_image(img).astype(np.float32)

    if dataset_stats is None:
        raise ValueError("dataset_stats is required for dataset z-score normalization")
    mean, std = dataset_stats
    return zscore_with_dataset_stats(img, mean, std).astype(np.float32)


def save_npz(out_npz_path: Path, image: np.ndarray, label: np.ndarray):
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, image=image.astype(np.float32), label=label.astype(np.int16))


def write_txt(txt_path: Path, stems, mode: str):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if mode == "overwrite" else "a"
    with open(txt_path, file_mode, encoding="utf-8") as f:
        for s in stems:
            f.write(f"{s}\n")


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    out_root = Path(args.out_root)

    if not args.no_masks and masks_dir is None:
        raise ValueError("--masks_dir is required unless --no_masks is set")

    out_npz_dir = out_root / "train_npz"
    out_list_dir = out_root / "lists" / args.dataset_name

    samples = collect_samples(images_dir, masks_dir, args.img_suffix, args.mask_suffix, args.no_masks)
    print(f"[INFO] found {len(samples)} samples")
    print("[INFO] use_mask_for_norm: [false, false, false]")

    label_mapping = parse_label_map(args.label_map)

    auto_mapping = None
    if (not args.no_masks) and (not args.keep_mask_values) and (label_mapping is None):
        all_vals = set()
        for _, mask_p, _ in samples:
            mask = np.array(Image.open(mask_p))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            all_vals.update(np.unique(mask).tolist())
        uniq = sorted(list(all_vals))
        auto_mapping = {v: i for i, v in enumerate(uniq)}
        print(f"[INFO] auto label mapping: {auto_mapping}")

    dataset_stats = None
    if args.norm == "zscore" and args.zscore_scope == "dataset":
        dataset_stats = compute_dataset_zscore_stats(samples)
        print("[INFO] computed dataset z-score stats (rgb=True)")

    generated_stems = []
    for img_p, mask_p, stem in samples:
        image = load_image(img_p)
        image = normalize_image(image, norm=args.norm, zscore_scope=args.zscore_scope, dataset_stats=dataset_stats)

        if args.no_masks:
            mask = np.zeros(image.shape[:2], dtype=np.int16)
        else:
            mask = np.array(Image.open(mask_p))
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            if label_mapping is not None:
                mask = remap_mask(mask, label_mapping)
            elif auto_mapping is not None:
                mask = remap_mask(mask, auto_mapping)
            else:
                mask = mask.astype(np.int16)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Shape mismatch for {stem}: image {image.shape}, mask {mask.shape}")

        save_npz(out_npz_dir / f"{stem}.npz", image, mask)
        generated_stems.append(stem)

    generated_stems = sorted(generated_stems)
    split_txt = out_list_dir / f"{args.target_split}.txt"
    write_txt(split_txt, generated_stems, args.write_mode)

    if args.all_txt:
        write_txt(out_list_dir / "all.txt", generated_stems, "overwrite")

    print(f"[DONE] generated npz: {len(generated_stems)} -> {out_npz_dir}")
    print(f"[DONE] wrote split file: {split_txt} (mode={args.write_mode})")
    if args.all_txt:
        print(f"[DONE] wrote all list: {out_list_dir / 'all.txt'}")


if __name__ == "__main__":
    main()
