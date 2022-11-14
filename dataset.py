import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


def get_augment(img_size):
    img_aug = {
        'train': transforms.Compose([
                transforms.ColorJitter(brightness=20, contrast=0.2, saturation=20., hue=[-0.5, 0.5]),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAffine(degrees=180, scale=(0.01, 0.2), shear=0.2, translate=(0.2, 0.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor()
                ]),
        'val': transforms.Compose([
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor()
        ])
    }
    return img_aug


class APTOSDataset(Dataset):
    def __init__(self, df_path, augmentation=None):
        self.df = pd.read_csv(df_path)
        self.len = len(self.df)
        self.labels = self.df.diagnosis.tolist()
        self.images = self.df.path.tolist()
        self.augmentation = augmentation

    def __getitem__(self, index):
        label = self.labels[index]
        path = self.images[index]
        img = Image.open(path)
        if self.augmentation is not None:
            img = self.augmentation(img)
        return img, label

    def __len__(self):
        return self.len


def get_dataloaders(train_df_path, val_df_path, img_size, batch_size):
    augment = get_augment(img_size)
    train_dataset = APTOSDataset(train_df_path, augmentation=augment['train'])
    val_dataset = APTOSDataset(val_df_path, augmentation=augment['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=12,
                                               drop_last=True
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=12,
                                              drop_last=False
                                             )
    return train_loader, val_loader

from pytorch_grad_cam import GradCAM
