import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 48,
    num_workers: int = 4,
    augment: bool = False,
    grayscale: bool = True
):
    """
    Returns a dict with DataLoaders for 'train', 'val', and 'test' based on the following structure:
    data_dir/
        train/0/, train/1/, …, train/6/
        val/0/, …
        test/0/, …

    Args:
      data_dir: path to the directory with train/, val/, test/ subfolders
      batch_size: batch size
      img_size: image size (crop/resize to square img_size x img_size)
      num_workers: number of workers for DataLoader
      augment: whether to enable augmentation (only for train)
      grayscale: whether to convert images to grayscale or keep RGB
    Returns:
      {"train": train_loader, "val": val_loader, "test": test_loader}
    """

    common_tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),            # z [0,255]→[0,1], float32
    ]
    
    if grayscale:
        common_tfms.insert(0, transforms.Grayscale(num_output_channels=1))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # to [-1,1]
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1,1]

    train_tfms = []
    if augment:
        train_tfms += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20, expand=False, center=None),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            # transforms.RandAugment(num_ops=3, magnitude=9),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ]                                    

    train_tfms += common_tfms + [normalize]

    val_tfms = common_tfms + [normalize]

    loaders = {}
    for split in ("train", "val", "test"):
        tfms = train_tfms if (split=="train" and augment) else val_tfms
        ds = datasets.ImageFolder(
            os.path.join(data_dir, split),
            transform=transforms.Compose(tfms)
        )
        shuffle = (split == "train")
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders[split] = loader

    return loaders