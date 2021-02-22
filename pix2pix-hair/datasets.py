import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def transform(img1, img2, image_size, model, device):
    resize = transforms.Resize(size=(image_size, image_size))
    img1 = resize(img1)
    img2 = resize(img2)
    img1, img2 = TF.to_tensor(img1), TF.to_tensor(img2)
    img1, img2 = TF.normalize(img1, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), TF.normalize(img2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img1, img2 = img1.to(device), img2.to(device)
    # mask
    mask1 = torch.argmax(model(img1.unsqueeze(0)), 1).type(torch.uint8).repeat(3, 1 ,1).to(device)
    mask2 = torch.argmax(model(img2.unsqueeze(0)), 1).type(torch.uint8).repeat(3, 1 ,1).to(device)
    return img1, mask1, img2, mask2


class ImageDataset(Dataset):
    def __init__(self, train, model, config, transform=transform):
        self.train = train
        self.model = model
        self.config = config
        self.transform = transform
        self.images = [os.path.join(config.image_path, img) for img in os.listdir(config.image_path)]
    
    def __len__(self):
        n = len(self.images)
        n1 = int(n * self.config.p)
        if self.train:
            return n1
        else:
            return n-n1

    def __getitem__(self, idx):
        n = len(self.images)
        n1 = int(n * self.config.p)
        if self.train:
            other = random.randint(0, n1)
            while other == idx:
                other = random.randint(0, n1)
            img1 = Image.open(self.images[idx])
            img2 = Image.open(self.images[other])
        else:
            other = random.randint(n1, n)
            while other == idx:
                other = random.randint(n1, n)
            img1 = Image.open(self.images[idx])
            img2 = Image.open(self.images[other])
        return self.transform(img1, img2, 
                self.config.image_size, self.model, self.config.device)


def get_loaders(model, config, transform=transform):
    train_dataset = ImageDataset(True, model, config, transform=transform)
    val_dataset = ImageDataset(False, model, config, transform=transform)
    return DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
