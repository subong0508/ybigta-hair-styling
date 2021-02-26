import os
import sys
import numpy as np

import torch
from torch.autograd import Variable
from PIL import Image
from configs import get_config
from utils import *
from datasets import get_loaders
from models import GeneratorUNet, Discriminator
from networks import Vgg16, ResNet18
from datasets import transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
    
def inference(config):
    hair_model, skin_model = load_model(config)

    #train_loader, val_loader = get_loaders(hair_model, skin_model, config)

    try:
        your_pic = Image.open(config.your_pic)
        celeb_pic = Image.open(config.celeb_pic)

    except:
        return
    
    your_pic,your_pic_mask,celeb_pic,celeb_pic_mask = DataLoader(transform(your_pic,celeb_pic,config.image_size,
                                                                    hair_model,skin_model,config.device))

    # Initialize
    vgg = Vgg16().to(config.device)
    resnet = ResNet18(requires_grad=True, pretrained=True).to(config.device)
    generator = GeneratorUNet().to(config.device)
    # discriminator = Discriminator().to(config.device)

    try:
        resnet.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch_%d_%s.pth' % (20, 'resnet'))))
        generator.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch_%d_%s.pth' % (20, 'generator'))))
    except OSError:
        print('Check if your pretrained weight is in the right place.')


    z1 = resnet(your_pic * your_pic_mask) #skin
    z2 = resnet(celeb_pic * celeb_pic_mask) #hair
    fake_im = generator(your_pic, z1, z2) # z1 is skin, z2 is hair


    images = [your_pic[0],celeb_pic[0],fake_im[0]]
    titles=['Your picture','Celebrity picture','Synthesized picture']

    fig, axes = plt.subplots(1,len(titles))
    for i in range(len(images)):
        im = images[i]
        im = im.data.cpu().numpy().transpose(1, 2, 0)
        im = (im + 1) / 2
        axes[i].imshow(im)
        axes[i].axis('off')
        axes[i].set_title(titles[i])
    
    plt.show()
   

if __name__ == "__main__":
    config = get_config()
    inference(config)