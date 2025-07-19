# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward,DWT1DForward

data_in_channel = 4
data_out_channel = 4
data_size = 512

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()        # 初始化离散小波变换，J=1表示变换的层数，mode='zero'表示填充模式，使用'Haar'小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')        # 定义卷积、批归一化和ReLU激活的顺序组合
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),  # 1x1卷积层，通道数由in_ch*4变为out_ch
            nn.BatchNorm2d(out_ch),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
    def forward(self, x):        # 对输入x进行离散小波变换，得到低频部分yL和高频部分yH
        yL, yH = self.wt(x)        # 提取高频部分的不同分量
        y_HL = yH[0][:, :, 0, ::]  # 水平高频
        y_LH = yH[0][:, :, 1, ::]  # 垂直高频
        # y_HH = yH[0][:, :, 2, ::]  # 对角高频        # 将低频部分和高频部分拼接在一起
        x = torch.cat([yL, y_HL, y_LH], dim=1)        # 通过卷积、批归一化和ReLU激活处理拼接后的特征
        x = self.conv_bn_relu(x)
        return x


class Down_wt_1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt_1d, self).__init__()
        # 初始化离散小波变换，J=1表示变换的层数，mode='zero'表示填充模式，使用'Haar'小波
        self.wt = DWT1DForward(J=1, mode='zero', wave='haar')
        # 定义卷积、批归一化和ReLU激活的顺序组合
        self.conv_bn_relu = nn.Sequential(
            nn.Conv1d(in_ch * 2, out_ch, kernel_size=1, stride=1),  # 1x1卷积层，通道数由in_ch*4变为out_ch
            nn.BatchNorm1d(out_ch),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
    def forward(self, x):        # 对输入x进行离散小波变换，得到低频部分yL和高频部分yH
        yL, yH = self.wt(x)        # 提取高频部分的不同分量
        y_HL = yH[0]  # 水平高频
        # y_LH = yH[0][:, :, 1, :]  # 垂直高频
        # y_HH = yH[0][:, :, 2, ::]  # 对角高频        # 将低频部分和高频部分拼接在一起
        x = torch.cat([yL, y_HL], dim=1)        # 通过卷积、批归一化和ReLU激活处理拼接后的特征
        x = self.conv_bn_relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv1d(inchannel, inchannel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(inchannel // ratio, inchannel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        # attention layers
        self.ca = ChannelAttention(outchannel)
        self.sa = SpatialAttention()

        # resnet layers
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.left = nn.Sequential(
                Down_wt_1d(inchannel, outchannel),
                nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(outchannel)
            )


            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # squeeze
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        # resnet
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class SCNet(nn.Module):
    def __init__(self, BasicBlock):
        super(SCNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(data_in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )

        self.layer1 = self.make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 3, stride=2)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 64, data_out_channel * data_size),
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, kernel_size=3, stride=1, padding=1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.reshape((out.size(0), data_out_channel, data_size))
        return out


def Wave_ResNet34():
    return SCNet(BasicBlock)

if __name__ == '__main__':

    model = Down_wt_1d(64,64)
    x = torch.randn((3, 64, 512))
    y = model(x)

