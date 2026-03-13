import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if image.ndim == 3:
        image = ndimage.rotate(image, angle, axes=(0, 1), order=1, reshape=False)
    else:
        image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_affine(image, label, max_rotate=15.0, max_scale=0.1, max_shift_ratio=0.05):
    h, w = image.shape[:2]
    angle = np.deg2rad(np.random.uniform(-max_rotate, max_rotate))
    scale = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)
    tx = np.random.uniform(-max_shift_ratio, max_shift_ratio) * w
    ty = np.random.uniform(-max_shift_ratio, max_shift_ratio) * h

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    transform = np.array([[scale * cos_a, -scale * sin_a], [scale * sin_a, scale * cos_a]])

    center = np.array([h / 2.0, w / 2.0])
    offset = center - transform @ center - np.array([ty, tx])

    if image.ndim == 3:
        out = np.empty_like(image)
        for c in range(image.shape[-1]):
            out[..., c] = ndimage.affine_transform(
                image[..., c], transform, offset=offset, order=1, mode="nearest"
            )
        image = out
    else:
        image = ndimage.affine_transform(image, transform, offset=offset, order=1, mode="nearest")

    label = ndimage.affine_transform(label, transform, offset=offset, order=0, mode="nearest")
    return image, label


def random_intensity(image, brightness=0.15, contrast=0.15, gamma_range=(0.8, 1.2), noise_std=0.02):
    image = image.astype(np.float32)
    mean = image.mean(axis=(0, 1), keepdims=True) if image.ndim == 3 else image.mean()
    image = (image - mean) * np.random.uniform(1.0 - contrast, 1.0 + contrast) + mean
    image = image + np.random.uniform(-brightness, brightness)

    im_min, im_max = image.min(), image.max()
    if im_max > im_min:
        image = (image - im_min) / (im_max - im_min)
        image = np.power(image, np.random.uniform(gamma_range[0], gamma_range[1]))
        image = image * (im_max - im_min) + im_min

    if np.random.random() < 0.5:
        image = image + np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)

    return image


def random_specular_reflection(image, p=0.25, max_spots=3):
    if image.ndim != 3 or image.shape[-1] < 3 or np.random.random() > p:
        return image

    h, w, _ = image.shape
    out = image.astype(np.float32).copy()

    yy, xx = np.ogrid[:h, :w]
    num_spots = np.random.randint(1, max_spots + 1)
    for _ in range(num_spots):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        rx = np.random.randint(max(4, w // 40), max(8, w // 12))
        ry = np.random.randint(max(4, h // 40), max(8, h // 12))

        mask = ((xx - cx) ** 2 / (rx ** 2 + 1e-6) + (yy - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1.0
        soft = np.exp(-(((xx - cx) ** 2) / (2 * (0.45 * rx) ** 2 + 1e-6) + ((yy - cy) ** 2) / (2 * (0.45 * ry) ** 2 + 1e-6)))
        intensity = np.random.uniform(0.35, 0.8)
        out = out + (intensity * soft[..., None] * mask[..., None])

    return out


def resize_and_to_tensor(image, label, output_size):
    x, y = image.shape[:2]
    if x != output_size[0] or y != output_size[1]:
        if image.ndim == 3:
            image = zoom(image, (output_size[0] / x, output_size[1] / y, 1), order=3)
        else:
            image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
        label = zoom(label, (output_size[0] / x, output_size[1] / y), order=0)

    if image.ndim == 2:
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    else:
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
    label = torch.from_numpy(label.astype(np.float32)).long()
    return {'image': image, 'label': label}


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < 0.35:
            image, label = random_rot_flip(image, label)
        if random.random() < 0.25:
            image, label = random_rotate(image, label)
        if random.random() < 0.30:
            image, label = random_affine(image, label)

        if random.random() < 0.70:
            image = random_intensity(image)
        image = random_specular_reflection(image, p=0.25)

        return resize_and_to_tensor(image, label, self.output_size)


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return resize_and_to_tensor(image, label, self.output_size)


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n').split(",")[0]

        # default: read npz (including test split)
        npz_path = os.path.join(self.data_dir, sample_name if sample_name.endswith(".npz") else sample_name + '.npz')
        if self.split in ["train", "val"] or os.path.exists(npz_path):
            data = np.load(npz_path)
            try:
                image, label = data['image'], data['label']
            except Exception:
                image, label = data['data'], data['seg']

            # test path expects an extra singleton channel before volume depth
            if self.split not in ["train", "val"]:
                if image.ndim == 3:
                    image = np.expand_dims(image, axis=0)
                if label.ndim == 3:
                    label = np.expand_dims(label, axis=0)
        else:
            # backward compatibility: fallback to legacy h5 test volume
            filepath = self.data_dir + "/{}.npy.h5".format(sample_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
