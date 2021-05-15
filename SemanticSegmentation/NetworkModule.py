import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        block_config = [3,4,6,3]
        img_size = 475
        img_size_8 = 60

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0],
            in_channels=128,
            mid_channel=64,
            out_channel=256,
            stride=1,
            dilation=1
        )
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1],
            in_channels=256,
            mid_channel=128,
            out_channel=512,
            stride=2,
            dilation=1
        )
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2],
            in_channels=512,
            mid_channel=256,
            out_channel=1024,
            stride=1,
            dilation=2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3],
            in_channels=1024,
            mid_channel=512,
            out_channel=2048,
            stride=1,
            dilation=4
        )

        self.pyramid_pooling = PyramidPooling(
            in_channels=2048,
            pool_sizes=[6, 3, 2, 1],
            height=img_size,
            width=img_size,
        )

        self.decode_feature = DecodePSPFeature(
            height=img_size,
            width=img_size,
            n_classes=n_classes
        )

        self.aux = AuxiliaryPSPlayers(
            in_channels=1024,
            height=img_size,
            width=img_size
        )

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)

        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs

class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3 , 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3 , 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3 , 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()

        self.add_module(
            "block1",
            bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)
        )

        for i in range(2, n_blocks + 1):
            self.add_module(
                "block" + str(i),
                bottleNeckPSP(
                    out_channels, mid_channels, stride, dilation)
            )

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()

        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.cb_3 = conv2DBatchNorm(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.cb_residual = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_1(x)
        conv = self.cbr_2(conv)
        conv = self.cb_3(conv)
        residual = self.cb_residual(x)
        return self.relu(conv + residual)

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super().__init__()

        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False
        )
        self.cb_3 = conv2DBatchNorm(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.cb_residual = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_1(x)
        conv = self.cbr_2(conv)
        conv = self.cb_3(conv)
        residual = x
        return self.relu(conv + residual)
