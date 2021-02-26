import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


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
    def __init__(self, in_size, out_size, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
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

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, z1, z2):
        # U-Net generator with skip connections from encoder to decoder
        # z1 = skin seg feature(512x1x1)
        # z2 = hair seg feature(512x1x1)
        d11 = self.down1(x) #64x128x128
        d21 = self.down2(d11) #128x64x64
        d31 = self.down3(d21) #256x32x32
        d41 = self.down4(d31) #512x16x16
        d51 = self.down5(d41) #512x8x8
        d61 = self.down6(d51) #512x4x4
        d71 = self.down7(d61) #512x2x2
        d81 = self.down8(d71) #512x1x1
        # skin + encoder 통과한 내 이미지
        d81 += z1
        u11 = self.up1(d81, d71) #512x2x2
        u21 = self.up2(u11, d61) #512x4x4
        u31 = self.up3(u21, d51) #512x8x8
        u41 = self.up4(u31, d41) #512x16x16
        u51 = self.up5(u41, d31) #256x32x32
        u61 = self.up6(u51, d21) #128x64x64
        u71 = self.up7(u61, d11) #64x128x128

        d12 = self.down1(x)
        d22 = self.down2(d12)
        d32 = self.down3(d22)
        d42 = self.down4(d32)
        d52 = self.down5(d42)
        d62 = self.down6(d52)
        d72 = self.down7(d62)
        d82 = self.down8(d72)
        # hair + encoder 통과한 내 이미지
        d82 += z2
        u12 = self.up1(d82, d72)
        u22 = self.up2(u12, d62)
        u32 = self.up3(u22, d52)
        u42 = self.up4(u32, d42)
        u52 = self.up5(u42, d32)
        u62 = self.up6(u52, d22)
        u72 = self.up7(u62, d12)

        u = (u71*0.2+u72*0.8)

        return self.final(u) #G(x,z1,z2) 생성 이미지


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
            # 두 이미지가 한번에 들어감 -> 채널 x 2
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