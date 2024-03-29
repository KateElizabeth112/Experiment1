# This file contains the UNet class and associated objects
import torch
import torch.nn as nn


# This block performs two convolutions followed by a ReLU
def double_conv(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.InstanceNorm2d(num_features=out_channels, affine=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
        nn.InstanceNorm2d(num_features=out_channels, affine=True),
        nn.LeakyReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, inChannels=1, outChannels=1, imgSize=512):
        super().__init__()

        self.inChannels = inChannels
        self.outChannels = outChannels

        self.dconv_down1 = double_conv(1, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dconv_down5 = double_conv(256, 512)

        # Extra layers to match nnUNet (only one extra layer as min feature map size is 4
        self.dconv_down6 = double_conv(512, 512)
        self.dconv_down7 = double_conv(512, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dropout = nn.Dropout2d(0.5)

        self.dconv_up6 = double_conv(512 + 512, 512)
        self.dconv_up5 = double_conv(512 + 512, 512)
        self.dconv_up4 = double_conv(256 + 512, 256)
        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(128 + 64, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, outChannels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #######   ENCODER ###############
        # Size is 512
        conv1 = self.dconv_down1(x)
        conv1 = self.dropout(conv1)
        x = self.maxpool(conv1)

        # Size is 256
        conv2 = self.dconv_down2(x)
        conv2 = self.dropout(conv2)
        x = self.maxpool(conv2)

        # Size is 128
        conv3 = self.dconv_down3(x)
        conv3 = self.dropout(conv3)
        x = self.maxpool(conv3)

        # Size is 64
        conv4 = self.dconv_down4(x)
        conv4 = self.dropout(conv4)
        x = self.maxpool(conv4)

        # Size is 32
        conv5 = self.dconv_down5(x)
        conv5 = self.dropout(conv5)
        x = self.maxpool(conv5)

        # Size is 16
        conv6 = self.dconv_down6(x)
        conv6 = self.dropout(conv6)
        x = self.maxpool(conv6)

        # Size is 8

        # Size is 4

        ######  MID-SECTION ######

        conv7 = self.dconv_down7(x)
        conv7 = self.dropout(conv7)

        ######  DECODER     ######

        deconv6 = self.upsample(conv7)
        deconv6 = torch.cat([deconv6, conv6], dim=1)
        deconv6 = self.dconv_up6(deconv6)
        deconv6 = self.dropout(deconv6)

        deconv5 = self.upsample(deconv6)
        deconv5 = torch.cat([deconv5, conv5], dim=1)
        deconv5 = self.dconv_up5(deconv5)
        deconv5 = self.dropout(deconv5)

        deconv4 = self.upsample(deconv5)
        deconv4 = torch.cat([deconv4, conv4], dim=1)
        deconv4 = self.dconv_up4(deconv4)
        deconv4 = self.dropout(deconv4)

        deconv3 = self.upsample(deconv4)
        deconv3 = torch.cat([deconv3, conv3], dim=1)
        deconv3 = self.dconv_up3(deconv3)
        deconv3 = self.dropout(deconv3)

        deconv2 = self.upsample(deconv3)
        deconv2 = torch.cat([deconv2, conv2], dim=1)
        deconv2 = self.dconv_up2(deconv2)
        deconv2 = self.dropout(deconv2)

        deconv1 = self.upsample(deconv2)
        deconv1 = torch.cat([deconv1, conv1], dim=1)
        deconv1 = self.dconv_up1(deconv1)
        deconv1 = self.dropout(deconv1)

        out = self.conv_last(deconv1)
        out_probs = self.softmax(out)

        return out_probs, deconv1
