import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 48,
    num_workers: int = 4,
    augment: bool = False
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
    Returns:
      {"train": train_loader, "val": val_loader, "test": test_loader}
    """

    common_tfms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),            # z [0,255]→[0,1], float32
    ]
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # do [-1,1]

    train_tfms = []
    if augment:
        train_tfms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
