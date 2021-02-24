import os
import sys
import numpy as np

import torch
from torch.autograd import Variable

from configs import get_config
from utils import *
from datasets import get_loaders
from models import GeneratorUNet, Discriminator
from networks import Vgg16, ResNet18


def main(config):
    model = load_model(config)
    train_loader, val_loader = get_loaders(model, config)
    
    # Make dirs
    if not os.path.exists(config.checkpoints):
        os.makedirs(config.checkpoints, exist_ok=True)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    # Loss Functions
    criterion_GAN = mse_loss

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, config.image_size // 2 ** 4, config.image_size // 2 ** 4)

    # Initialize
    vgg = Vgg16().to(config.device)
    resnet = ResNet18(requires_grad=True, pretrained=True).to(config.device)
    generator = GeneratorUNet().to(config.device)
    # discriminator = Discriminator().to(config.device)

    if config.epoch != 0:
    # Load pretrained models
        resnet.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch:%d_%s.pth' % (config.epoch, 'resnet'))))
        generator.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch:%d_%s.pth' % (config.epoch, 'generator'))))
        # discriminator.load_state_dict(torch.load(os.path.join(config.checkpoints, 'epoch:%d_%s.pth' % (config.epoch, 'discriminator'))))
    else:
        # Initialize weights
        # resnet.apply(weights_init_normal)
        generator.apply(weights_init_normal)
        # discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    # ----------
    #  Training
    # ----------

    resnet.train()
    generator.train()
    # discriminator.train()
    for epoch in range(config.epoch, config.n_epochs):
        for i, (im1, m1, im2, m2) in enumerate(train_loader):
            assert im1.size(0) == im2.size(0)
            valid = Variable(torch.Tensor(np.ones((im1.size(0), *patch))).to(config.device), requires_grad=False)
            fake = Variable(torch.Tensor(np.ones((im1.size(0), *patch))).to(config.device), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_resnet.zero_grad()
            optimizer_G.zero_grad()

            # GAN loss
            z = resnet(im2 * m2)
            fake_im = generator(im1, im2, z)
            # pred_fake = discriminator(fake_im, im2)
            # loss_GAN = config.lambda_gan * criterion_GAN(pred_fake, valid)

            # Hair, Face loss
            fake_m2 = torch.argmax(model(fake_im), 1).unsqueeze(1).type(torch.uint8).repeat(1, 3, 1, 1).to(config.device)
            if 0.5 * torch.sum(m1)  <= torch.sum(fake_m2) <= 1.5 * torch.sum(m1):
                hair_loss = config.lambda_style * calc_style_loss(fake_im * fake_m2, im2 * m2, vgg) + calc_content_loss(fake_im * fake_m2, im2 * m2, vgg)
                face_loss = calc_content_loss(fake_im, im1, vgg)
            else:
                hair_loss = config.lambda_style * calc_style_loss(fake_im * m1, im2 * m2, vgg) + calc_content_loss(fake_im * m1, im2 * m2, vgg)
                face_loss = calc_content_loss(fake_im, im1, vgg)
            hair_loss *= config.lambda_hair
            face_loss *= config.lambda_face

            # Total loss
            loss = hair_loss + face_loss
            
            loss.backward()
            optimizer_resnet.step()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # optimizer_D.zero_grad()

            # # Real loss
            # pred_real = discriminator(im1, im2)
            # loss_real = criterion_GAN(pred_real, valid)
            # # Fake loss
            # pred_fake = discriminator(fake_im.detach(), im2)
            # loss_fake = criterion_GAN(pred_fake, fake)
            # # Total loss
            # loss_D = 0.5 * (loss_real + loss_fake)

            # loss_D.backward()
            # optimizer_D.step()

            if i % config.sample_interval == 0:
                msg = "Train || hair loss: %.6f, face loss: %.6f, loss: %.6f\n" % \
                    (hair_loss.item(), face_loss.item(), loss.item())
                sys.stdout.write("Epoch: %d || Batch: %d\n" % (epoch, i))
                sys.stdout.write(msg)
                fname = os.path.join(config.save_path, "Train_Epoch:%d_Batch:%d.png" % (epoch, i))
                sample_images([im1[0], im2[0], fake_im[0]], ["img1", "img2", "img1+img2"], fname)
                for j, (im1, m1, im2, m2) in enumerate(val_loader):
                    with torch.no_grad():
                        valid = Variable(torch.Tensor(np.ones((im1.size(0), *patch))).to(config.device), requires_grad=False)
                        fake = Variable(torch.Tensor(np.ones((im1.size(0), *patch))).to(config.device), requires_grad=False)

                        # GAN loss
                        z = resnet(im2 * m2)
                        fake_im = generator(im1, im2, z)
                        # pred_fake = discriminator(fake_im, im2)
                        # loss_GAN = config.lambda_gan * criterion_GAN(pred_fake, valid)

                        # Hair, Face loss
                        fake_m2 = torch.argmax(model(fake_im), 1).unsqueeze(1).type(torch.uint8).repeat(1, 3, 1, 1).to(config.device)
                        if 0.5 * torch.sum(m1)  <= torch.sum(fake_m2) <= 1.5 * torch.sum(m1):
                            hair_loss = config.lambda_style * calc_style_loss(fake_im * fake_m2, im2 * m2, vgg) + calc_content_loss(fake_im * fake_m2, im2 * m2, vgg)
                            face_loss = calc_content_loss(fake_im, im1, vgg)
                        else:
                            hair_loss = config.lambda_style * calc_style_loss(fake_im * m1, im2 * m2, vgg) + calc_content_loss(fake_im * m1, im2 * m2, vgg)
                            face_loss = calc_content_loss(fake_im, im1, vgg)
                        hair_loss *= config.lambda_hair
                        face_loss *= config.lambda_face

                        # Total loss
                        loss = hair_loss + face_loss
                        
                        # loss_G.backward()
                        # optimizer_resnet.step()
                        # optimizer_G.step()

                        # ---------------------
                        #  Train Discriminator
                        # ---------------------

                        # Real loss
                        # pred_real = discriminator(im1, im2)
                        # loss_real = criterion_GAN(pred_real, valid)
                        # # Fake loss
                        # pred_fake = discriminator(fake_im.detach(), im2)
                        # loss_fake = criterion_GAN(pred_fake, fake)
                        # # Total loss
                        # loss_D = 0.5 * (loss_real + loss_fake)

                        msg = "Validation || hair loss: %.6f, face loss: %.6f, loss: %.6f\n" % \
                                (hair_loss.item(), face_loss.item(), loss.item())
                        sys.stdout.write(msg)
                        fname = os.path.join(config.save_path, "Validation_Epoch:%d_Batch:%d.png" % (epoch, i))
                        sample_images([im1[0], im2[0], fake_im[0]], ["img1", "img2", "img1+img2"], fname)
                        break

        if epoch % config.checkpoint_interval == 0:
            models = [resnet, generator]
            fnames = ['resnet', 'generator']
            fnames = [os.path.join(config.checkpoints, 'epoch:%d_%s.pth' % (epoch, s)) for s in fnames]
            save_weights(models, fnames)


if __name__ == "__main__":
    config = get_config()
    main(config)