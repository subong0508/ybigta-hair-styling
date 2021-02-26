import os
import matplotlib.pyplot as plt
import torch
from hairsegmentation.model.transfer_model import MobileHairNet


mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()


def load_model(config):
    hair_model = MobileHairNet().to(config.device)
    hair_model_name = os.path.join(config.model_hair_path, config.model_hair_weights)
    skin_model = MobileHairNet().to(config.device)
    skin_model_name = os.path.join(config.model_skin_path, config.model_skin_weights)
    print("Loading: %s..." % hair_model_name)
    print("Loading: %s..." % skin_model_name)
    hair_model.load_state_dict(torch.load(hair_model_name, map_location=config.device))
    skin_model.load_state_dict(torch.load(skin_model_name, map_location=config.device))
    if config.freeze:
        for param in hair_model.parameters():
            param.requires_grad = False
        for param in skin_model.parameters():
            param.requires_grad = False
        hair_model.eval()
        skin_model.eval()
    print("Done.")
    return hair_model, skin_model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def calc_style_loss(img1, img2, vgg):
    """
    Params
    @img1, img2: images
    @vgg: pretrained vgg16 network
    Return: style loss
    """
    if img1.shape[1] != 3:
        img1 = img1.repeat(1, 3, 1, 1)
    if img2.shape[1] != 3:
        img2 = img2.repeat(1, 3, 1, 1)
    img1, img2 = map(lambda x: (x+1) * 0.5 * 255, [img1, img2])
    style1 = vgg(normalize_batch(img1))
    style2 = vgg(normalize_batch(img2))
    gram_style1 = [gram_matrix(x) for x in style1]
    gram_style2 = [gram_matrix(y) for y in style2]
    style_loss = 0
    for st1, st2 in zip(gram_style1, gram_style2):
        style_loss += mse_loss(st1, st2)
    del style1, style2, gram_style1, gram_style2
    return style_loss


def calc_content_loss(img1, img2, vgg):
    """
    Params
    @img1, img2: images
    @vgg: pretrained vgg16 network
    Return: content loss
    """
    if img1.shape[1] != 3:
        img1 = img1.repeat(1, 3, 1, 1)
    if img2.shape[1] != 3:
        img2 = img2.repeat(1, 3, 1, 1)
    img1, img2 = map(lambda x: (x+1) * 0.5 * 255, [img1, img2])
    style1 = vgg(normalize_batch(img1))
    style2 = vgg(normalize_batch(img2))
    return mse_loss(style1.relu2_2, style2.relu2_2)


def sample_images(images, titles, fname):
    assert len(images) == len(titles)
    fig, axes = plt.subplots(1, len(titles))
    for i in range(len(images)):
        im = images[i]
        im = im.data.cpu().numpy().transpose(1, 2, 0)
        im = (im + 1) / 2
        axes[i].imshow(im)
        axes[i].axis('off')
        axes[i].set_title(titles[i])
    plt.savefig(fname)
    #plt.show()
    


def save_weights(models, fnames):
    for m, f in zip(models, fnames):
        torch.save(m.state_dict(), f)