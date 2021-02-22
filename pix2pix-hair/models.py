import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down11 = UNetDown(in_channels, 64, normalize=False)
        self.down21 = UNetDown(64, 128)
        self.down31 = UNetDown(128, 256)
        self.down41 = UNetDown(256, 512, dropout=0.5)
        self.down51 = UNetDown(512, 512, dropout=0.5)
        self.down61 = UNetDown(512, 512, dropout=0.5)
        self.down71 = UNetDown(512, 512, dropout=0.5)
        self.down81 = UNetDown(512, 512, dropout=0.5)
        self.up11 = UNetUp(512, 512, dropout=0.5)
        self.up21 = UNetUp(1024, 512, dropout=0.5)
        self.up31 = UNetUp(1024, 512, dropout=0.5)
        self.up41 = UNetUp(1024, 512, dropout=0.5)
        self.up51 = UNetUp(1024, 256)
        self.up61 = UNetUp(512, 128)
        self.up71 = UNetUp(256, 64)

        self.down12 = UNetDown(in_channels, 64, normalize=False)
        self.down22 = UNetDown(64, 128)
        self.down32 = UNetDown(128, 256)
        self.down42 = UNetDown(256, 512, dropout=0.5)
        self.down52 = UNetDown(512, 512, dropout=0.5)
        self.down62 = UNetDown(512, 512, dropout=0.5)
        self.down72 = UNetDown(512, 512, dropout=0.5)
        self.down82 = UNetDown(512, 512, dropout=0.5)
        self.up12 = UNetUp(512, 512, dropout=0.5)
        self.up22 = UNetUp(1024, 512, dropout=0.5)
        self.up32 = UNetUp(1024, 512, dropout=0.5)
        self.up42 = UNetUp(1024, 512, dropout=0.5)
        self.up52 = UNetUp(1024, 256)
        self.up62 = UNetUp(512, 128)
        self.up72 = UNetUp(256, 64)

        self.final1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        self.final2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x1, x2, z):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down11(x1)
        d2 = self.down21(d1)
        d3 = self.down31(d2)
        d4 = self.down41(d3)
        d5 = self.down51(d4)
        d6 = self.down61(d5)
        d7 = self.down71(d6)
        d8 = self.down81(d7)
        # Hair Feature map
        d8 += z
        u1 = self.up11(d8, d7)
        u2 = self.up21(u1, d6)
        u3 = self.up31(u2, d5)
        u4 = self.up41(u3, d4)
        u5 = self.up51(u4, d3)
        u6 = self.up61(u5, d2)
        u71 = self.up71(u6, d1)
        
        d1 = self.down12(x2)
        d2 = self.down22(d1)
        d3 = self.down32(d2)
        d4 = self.down42(d3)
        d5 = self.down52(d4)
        d6 = self.down62(d5)
        d7 = self.down72(d6)
        d8 = self.down82(d7)
        u1 = self.up12(d8, d7)
        u2 = self.up22(u1, d6)
        u3 = self.up32(u2, d5)
        u4 = self.up42(u3, d4)
        u5 = self.up52(u4, d3)
        u6 = self.up62(u5, d2)
        u72 = self.up72(u6, d1)

        out = (self.final1(u71) + self.final2(u72)) / 2

        return out


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)