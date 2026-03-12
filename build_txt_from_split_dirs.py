#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate train/val/test txt files from existing split folders of NPZ files."""

import argparse
from pathlib import Path


def remove_last_underscore_suffix(stem: str) -> str:
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def list_stems(npz_dir: Path, remove_last_underscore: bool = False):
    if not npz_dir.exists():
        raise FileNotFoundError(f"Directory not found: {npz_dir}")
    stems = sorted([p.stem for p in npz_dir.glob("*.npz")])
    if remove_last_underscore:
        stems = [remove_last_underscore_suffix(s) for s in stems]
    return stems


def write_txt(txt_path: Path, stems):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for stem in stems:
            f.write(f"{stem}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train.txt / val.txt / test.txt from existing NPZ split folders"
    )
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing training NPZ files")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory containing validation NPZ files")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test NPZ files")
    parser.add_argument("--out_list_dir", type=str, required=True, help="Output directory for train.txt/val.txt/test.txt")
    parser.add_argument("--allow_empty", action="store_true", help="Allow empty split directories")
    parser.add_argument(
        "--remove_last_underscore",
        action="store_true",
        help="Remove the last '_' suffix from each NPZ stem before writing txt",
    )
    args = parser.parse_args()

    train_stems = list_stems(Path(args.train_dir), remove_last_underscore=args.remove_last_underscore)
    val_stems = list_stems(Path(args.val_dir), remove_last_underscore=args.remove_last_underscore)
    test_stems = list_stems(Path(args.test_dir), remove_last_underscore=args.remove_last_underscore)

    if not args.allow_empty:
        if len(train_stems) == 0:
            raise ValueError("No NPZ found in train_dir")
        if len(val_stems) == 0:
            raise ValueError("No NPZ found in val_dir")
        if len(test_stems) == 0:
            raise ValueError("No NPZ found in test_dir")

    out_dir = Path(args.out_list_dir)
    write_txt(out_dir / "train.txt", train_stems)
    write_txt(out_dir / "val.txt", val_stems)
    write_txt(out_dir / "test.txt", test_stems)

    print(f"[DONE] train: {len(train_stems)} -> {out_dir / 'train.txt'}")
    print(f"[DONE] val: {len(val_stems)} -> {out_dir / 'val.txt'}")
    print(f"[DONE] test: {len(test_stems)} -> {out_dir / 'test.txt'}")


if __name__ == "__main__":
    main()
