from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ResNet18(torch.nn.Module):
    def __init__(self, requires_grad=True, pretrained=False):
        super(ResNet18, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == "__main__":
    from configs import get_config
    from models import GeneratorUNet

    config = get_config()
    resnet = ResNet18(False).to(config.device)
    t = torch.randn((64, 3, 256, 256)).to(config.device)
    # 64 x 512 x 1 x 1
    z = resnet(t)
    gen = GeneratorUNet().to(config.device)
    with torch.no_grad():
        print(gen(t, z).shape)