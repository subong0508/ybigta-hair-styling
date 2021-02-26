import os
import random
from random import random as r
from utils import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def transform(img1, img2, image_size, hair_model, skin_model, device):
    resize = transforms.Resize(size=(image_size, image_size))
    img1 = resize(img1)
    img2 = resize(img2)

    if r() > 0.5:
        img1 = TF.vflip(img1)
        img2 = TF.vflip(img2)

    if r() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)

    if r() > 0.5:
        angle = r() * 12 - 6
        img1 = TF.rotate(img1, angle)

    if r() > 0.5:
        angle = r() * 12 - 6
        img2 = TF.rotate(img2, angle)

    img1, img2 = TF.to_tensor(img1), TF.to_tensor(img2)
    img1, img2 = TF.normalize(img1, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), TF.normalize(img2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img1, img2 = img1.to(device), img2.to(device)

    # mask
    mask1 = torch.argmax(skin_model(img1.unsqueeze(0)), 1).type(torch.uint8).repeat(3, 1 ,1).to(device)
    mask2 = torch.argmax(hair_model(img2.unsqueeze(0)), 1).type(torch.uint8).repeat(3, 1 ,1).to(device)
    return img1, mask1, img2, mask2


class ImageDataset(Dataset):
    def __init__(self, train, hair_model, skin_model, config, transform=transform):
        self.train = train
        self.skin_model = skin_model
        self.hair_model = hair_model
        self.config = config
        self.transform = transform
        self.images = [os.path.join(config.image_path, img) for img in os.listdir(config.image_path)]
        random.shuffle(self.images)
    
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
            other = random.randint(0, n1-1)
            while other == idx:
                other = random.randint(0, n1-1)
            img1 = Image.open(self.images[idx])
            img2 = Image.open(self.images[other])
        else:
            other = random.randint(0, n-n1-1)
            while other == idx:
                other = random.randint(0, n-n1-1)
            img1 = Image.open(self.images[n1+idx])
            img2 = Image.open(self.images[n1+other])
        return self.transform(img1, img2, 
                self.config.image_size, self.hair_model, self.skin_model, self.config.device)


def get_loaders(hair_model, skin_model, config, transform=transform):
    train_dataset = ImageDataset(True, hair_model, skin_model, config, transform=transform)
    val_dataset = ImageDataset(False, hair_model, skin_model, config, transform=transform)
    return DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    
def save_images(im1, im2, titles, filename, *, mask=False):
    im1 = im1.data.cpu().numpy().transpose(1, 2, 0)
    im1 = (im1 + 1) / 2
    im2 = im2.data.cpu().numpy().transpose(1, 2, 0)
    im2 = (im2 + 1) / 2
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.title(titles[0])
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.title(titles[1])
    if not mask:
        plt.savefig(f'samples/images/{filename}.png')
    else:
        plt.savefig(f'samples/masks/{filename}.png')

if __name__ == "__main__":
    from configs import get_config
    from utils import *

    config = get_config()
    hair_model, skin_model = load_model(config)
    train, val = get_loaders(hair_model, skin_model, config)
    print(len(train), len(val))
    for i, (im1, m1, im2, m2) in enumerate(train):
        print("Train: %d" % i)
    for i, (im1, m1, im2, m2) in enumerate(val):
        print("Validation: %d" % i)