import os
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from configs import get_config
from utils import * 
from datasets import transform
from models import GeneratorUNet, Discriminator
from networks import Vgg16, ResNet18


def main(config):
    model = load_model(config)
    your_pic = config.your_pic
    celeb_pic = config.celeb_pic
    try:
        your_pic = Image.open(your_pic)
        celeb_pic = Image.open(celeb_pic)
    except Exception:
        print("Provide correct paths!")
        return
    
    im1, m1, im2, m2 = DataLoader(transform(your_pic, celeb_pic, config.image_size, 
                                            model, config.device))
    im1, m1, im2, m2 = map(lambda x: x.repeat(2, 1, 1, 1), [im1, m1, im2, m2])

    # Initialize
    resnet = ResNet18(requires_grad=True, pretrained=True).to(config.device)
    generator = GeneratorUNet().to(config.device) 

    try:
        resnet.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch_%d_%s.pth' % (config.inf_epoch, 'resnet')), map_location=config.device))
        generator.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch_%d_%s.pth' % (config.inf_epoch, 'generator')), map_location=config.device))
    except OSError:
        print('Check if your pretrained weight is in the right place.')
                                     
    with torch.no_grad():
        resnet.eval()
        generator.eval()
        
        z = resnet(im2 * m2)
        fake_im = generator(im1, im2, z)
        images = [im1[0], im2[0], fake_im[0]]
        titles=['Your picture', 'Celebrity picture', 'Synthesized picture']

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
    main(config)