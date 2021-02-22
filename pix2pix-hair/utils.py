import os
import matplotlib.pyplot as plt
import torch
from hair_segmentation.model.transfer_model import MobileHairNet


mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()


def load_model(config):
    model = MobileHairNet().to(config.device)
    model_name = os.path.join(config.model_path, config.model_weights)
    print("Loading: %s..." % model_name)
    model.load_state_dict(torch.load(model_name, map_location=config.device))
    if config.freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    print("Done.")
    return model


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
    plt.show()
    plt.savefig(fname)


def save_weights(models, fnames):
    for m, f in zip(models, fnames):
        torch.save(m.state_dict(), f)