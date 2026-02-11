import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, Sampler
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np


class CameraDataset(Dataset):
    def __init__(self, viewpoint_stack, white_background, data_loader):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        self.data_loader = data_loader

    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]

        if self.data_loader:
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + self.bg * (
                    1 - norm_data[:, :, 3:4]
                )
                image_load = Image.fromarray(
                    np.array(arr * 255.0, dtype=np.uint8), "RGB"
                )
                resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
                viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
                if resized_image_rgb.shape[1] == 4:
                    gt_alpha_mask = resized_image_rgb[3:4, ...]
                    viewpoint_image *= gt_alpha_mask
                else:
                    viewpoint_image *= torch.ones(
                        (1, viewpoint_cam.image_height, viewpoint_cam.image_width)
                    )
        else:
            viewpoint_image = viewpoint_cam.original_image

        return viewpoint_image, viewpoint_cam

    def __len__(self):
        return len(self.viewpoint_stack)


class InfiniteSampler(Sampler[int]):
    def __init__(self, dataset_len: int, shuffle: bool = True, seed: int = 42):
        self.n = dataset_len
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                # new random order each pass, generated cheap & fast
                idx = torch.randperm(self.n, generator=g)
            else:
                idx = torch.arange(self.n)
            # yield indices continuously; no epoch boundary for DataLoader
            for i in idx.tolist():
                yield i

    def __len__(self):
        # Make it "practically infinite" so DataLoader never stops.
        return 2**31
