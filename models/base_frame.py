import torch
import torch.nn as nn

class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        x, identity = x
        x = self.relu(self.drop(self.conv1(x)))
        x = self.relu(self.drop(self.conv1(x)))
        x = self.relu(self.conv2(x))
        out = torch.cat((x, identity), dim=1)
        # 按照通道维度做拼接实现residual
        return out, identity

class SEMInet(nn.Module):
    def __init__(self, num_layers=3):
        super(SEMInet, self).__init__()
        self.input_step =  nn.Conv2d(in_channels=3, out_channels=64,
                                    kernel_size=3, stride=1, padding=1, bias=False)
        self.output_step = nn.Conv2d(in_channels=64, out_channels=3,
                                    kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.baseblock = self.StackBlock(BaseBlock, num_layers)
    def StackBlock(self, block, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.relu(self.input_step(x))

        out, _ = self.baseblock((x, identity))

        out = self.output_step(out)
        return out
