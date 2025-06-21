import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsImageFolder(datasets.ImageFolder):
    """A drop-in replacement for ImageFolder that applies Albumentations."""
    def __init__(self, root, transform=None, grayscale=False):
        super().__init__(root, transform=None)
        self.alb_transform = transform
        self.grayscale = grayscale

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path)
        img = img.convert("L" if self.grayscale else "RGB")
        img = np.array(img)
        if self.alb_transform:
            img = self.alb_transform(image=img)["image"]
        return img, target

def make_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 48,
    num_workers: int = 4,
    augment: bool = False,
    grayscale: bool = True
):    
    norm_mean = [0.5] if grayscale else [0.5, 0.5, 0.5]
    norm_std  = [0.5] if grayscale else [0.5, 0.5, 0.5]

    train_alb = A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), ratio=(0.9,1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05),
            A.GridDistortion(num_steps=5, distort_limit=0.05),
            A.PiecewiseAffine(scale=(0.02,0.05))
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomSnow(p=0.1),
        A.CoarseDropout(max_holes=1, max_height=int(img_size*0.2), max_width=int(img_size*0.2),
                        fill_value=norm_mean if grayscale else norm_mean + [0], p=0.3),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()
    ])

    val_alb = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()
    ])

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = (split == "train" and augment)
        transform = train_alb if is_train else val_alb
        ds = AlbumentationsImageFolder(
            os.path.join(data_dir, split),
            transform=transform,
            grayscale=grayscale
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split=="train"),
            num_workers=num_workers,
            pin_memory=True
        )

    return loaders
