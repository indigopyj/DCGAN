import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from layer import *


## DCGAN 네트워크
class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(DCGAN, self).__init__()

        # input : 1x100 linear vector == 1x1x100 matrix
        # output : 64x64x3
        self.dec1 = DECBR2d(1 * in_channels, 8 * nker, kernel_size=4, stride=1, padding=0, norm=norm, relu=0.0,
                            bias=False)
        self.dec2 = DECBR2d(8 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec3 = DECBR2d(4 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec4 = DECBR2d(2 * nker, 1 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec5 = DECBR2d(1 * nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None,
                            bias=False)

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = torch.tanh(x)

        return x



# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(DCGAN, self).__init__()

        # input : 64x64x3 matrix
        # output : 1x1x1 scalar value
        # output >= 0.5 : Real image
        # otherwise : fake image
        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = torch.sigmoid(x)

        return x

## Unet 네트워크 구축하기

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker, norm="bnorm", learning_type="plain"):  # Unet 정의할때 필요한 레이어 생성
        super(UNet, self).__init__()

        self.learning_type = learning_type

        # contracting path (encoder)
        self.enc1_1 = CBR2d(in_channels=in_channels, out_channels=nker, norm=norm)
        self.enc1_2 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker, norm=norm)

        # expansive path (decoder)
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker, norm=norm)

        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker, kernel_size=2, stride=2)

        self.dec4_2 = CBR2d(in_channels=2 * 8 * nker, out_channels=8 * nker, norm=norm)
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker, norm=norm)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker, kernel_size=2, stride=2)

        self.dec3_2 = CBR2d(in_channels=2 * 4 * nker, out_channels=4 * nker, norm=norm)
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker, norm=norm)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker, kernel_size=2, stride=2)

        self.dec2_2 = CBR2d(in_channels=2 * 2 * nker, out_channels=2 * nker, norm=norm)
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=nker, norm=norm)

        self.unpool1 = nn.ConvTranspose2d(in_channels=nker, out_channels=nker, kernel_size=2, stride=2)

        self.dec1_2 = CBR2d(in_channels=2 * nker, out_channels=nker, norm=norm)
        self.dec1_1 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=1)

    def forward(self, x):  # layer 연결
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # dim=[0:batch, 1:channel, 2:height, 3:width]
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            # residual learning: net이 input과 label의 '차이'만을 학습할 수 있도록함 - regression task에서 사용.
            x = self.fc(dec1_1) + x

        return x


## Autoencoder 네트워크 구축하기

class Hourglass(nn.Module):
    def __init__(self, in_channels, out_channels, nker, norm="bnorm", learning_type="plain"):  # Unet 정의할때 필요한 레이어 생성
        super(Hourglass, self).__init__()

        self.learning_type = learning_type

        # contracting path (encoder)
        self.enc1_1 = CBR2d(in_channels=in_channels, out_channels=nker, norm=norm)
        self.enc1_2 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker, norm=norm)

        # expansive path (decoder)
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker, norm=norm)

        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker, kernel_size=2, stride=2)

        self.dec4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker, norm=norm)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker, kernel_size=2, stride=2)

        self.dec3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker, norm=norm)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker, kernel_size=2, stride=2)

        self.dec2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=nker, norm=norm)

        self.unpool1 = nn.ConvTranspose2d(in_channels=nker, out_channels=nker, kernel_size=2, stride=2)

        self.dec1_2 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)
        self.dec1_1 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=1)

    def forward(self, x):  # layer 연결
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        # cat4 = torch.cat((unpool4, enc4_2), dim=1)  # dim=[0:batch, 1:channel, 2:height, 3:width]
        cat4 = unpool4
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        # cat3 = torch.cat((unpool3, enc3_2), dim=1)
        cat3 = unpool3
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        # cat2 = torch.cat((unpool2, enc2_2), dim=1)
        cat2 = unpool2
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        # cat1 = torch.cat((unpool1, enc1_2), dim=1)
        cat1 = unpool1
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            # residual learning: net이 input과 label의 '차이'만을 학습할 수 있도록함 - regression task에서 사용.
            x = self.fc(dec1_1) + x

        return x


class SRResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm", learning_type="plain", nblk=16):
        super(SRResNet, self).__init__()

        self.learning_type = learning_type

        # padding = kernel / 2
        self.enc = CBR2d(in_channels, nker, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=0.0)

        res = []

        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]

        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1,
                          padding=1)]  # out_channels에 4곱한 이유 : 채널사이즈가 64->256으로 바뀌었기 때문
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.ReLU()]

        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1,
                          padding=1)]  # out_channels에 4곱한 이유 : 채널사이즈가 64->256으로 바뀌었기 때문
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]

        self.ps2 = nn.Sequential(*ps2)

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.enc(x)
        x0 = x
        x = self.res(x)
        x = self.dec(x)
        x = x0 + x

        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm", learning_type="plain", nblk=16):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels=in_channels, out_channels=nker, kernel_size=3, stride=1, padding=1, bias=True,
                         norm=None, relu=0.0)

        res = []

        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]

        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x
