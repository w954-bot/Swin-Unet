#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Split existing NPZ files into train/val/test (default 8:1:1), write txt lists,
and move files into target folders.
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_ratio(text: str):
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("--ratio must be like 8:1:1")
    nums = [float(x) for x in parts]
    if any(x < 0 for x in nums) or sum(nums) == 0:
        raise ValueError("ratio values must be non-negative and not all zero")
    s = sum(nums)
    return [x / s for x in nums]


def write_txt(path: Path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{p.stem}\n")


def move_files(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dst_dir / src.name
        if dst.exists():
            raise FileExistsError(f"Target already exists: {dst}")
        shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, type=str, help="Folder containing existing .npz files")
    parser.add_argument("--out_root", required=True, type=str, help="Output root for moved files and lists")
    parser.add_argument("--dataset_name", default="MyDataset", type=str, help="Name under out_root/lists/")
    parser.add_argument("--ratio", default="8:1:1", type=str, help="Split ratio train:val:test")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--train_subdir", default="train_npz", type=str)
    parser.add_argument("--val_subdir", default="val_npz", type=str)
    parser.add_argument("--test_subdir", default="test_npz", type=str)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_root = Path(args.out_root)
    list_dir = out_root / "lists" / args.dataset_name

    files = sorted(src_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {src_dir}")

    train_r, val_r, test_r = parse_ratio(args.ratio)

    rnd = random.Random(args.seed)
    rnd.shuffle(files)

    n = len(files)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    # remainder goes to test to keep total exact
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    assert len(test_files) == n_test

    train_dir = out_root / args.train_subdir
    val_dir = out_root / args.val_subdir
    test_dir = out_root / args.test_subdir

    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)

    write_txt(list_dir / "train.txt", train_files)
    write_txt(list_dir / "val.txt", val_files)
    write_txt(list_dir / "test.txt", test_files)

    print(f"[DONE] total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    print(f"[DONE] moved to: {train_dir}, {val_dir}, {test_dir}")
    print(f"[DONE] lists: {list_dir / 'train.txt'}, {list_dir / 'val.txt'}, {list_dir / 'test.txt'}")


if __name__ == "__main__":
    main()
