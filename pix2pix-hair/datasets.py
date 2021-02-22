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
    mask1 = torch.argmax(model(img1.unsqueeze(0).to(device)).squeeze(0), 0).type(torch.uint8).to(device)
    mask1 = img1 * mask1
    mask2 = torch.argmax(model(img2.unsqueeze(0).to(device)).squeeze(0), 0).type(torch.uint8).to(device)
    mask2 = img2 * mask2
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
    from utils import load_model
    from networks import Vgg16
    from utils import calc_content_loss, calc_style_loss

    import warnings
    import numpy as np
    import matplotlib.pyplot as plt

    warnings.filterwarnings('ignore')

    config = get_config()
    model = load_model(config)
    vgg = Vgg16(requires_grad=False).to(config.device)

    dataset = ImageDataset(model, config)
    dataloader = DataLoader(dataset, shuffle=False)
    for i, (im1, m1, im2, m2) in enumerate(dataloader):
        s_loss = calc_style_loss(im1, im2, vgg)
        c_loss = calc_content_loss(im1, im2, vgg)
        f_name = f'{1e5 * s_loss:.6f}'
        im1.squeeze_(0)
        im2.squeeze_(0)
        save_images(im1, im2, [f"style:{1e5 * s_loss:.3f}", f"content:{c_loss:.3f}"], f_name)
        s_loss = calc_style_loss(m1, m2, vgg)
        c_loss = calc_content_loss(m1, m2, vgg)
        m1.squeeze_(0)
        m2.squeeze_(0)
        f_name = f'{1e5 * s_loss:.6f}'
        save_images(m1, m2, [f"style:{1e5 * s_loss:.3f}", f"content:{c_loss:.3f}"], f_name, mask=True)
        if i == 100:
            break