import torch.nn as nn


class AsymmetricBottleNeck(nn.Module):
    expansion = 4
    def __init__(self, opt, in_channels, out_channels, stride=1, dir=-1):
        super().__init__()
        self.opt = opt
        # downsampling
        if stride != 1:
            asy_kernel = 3
            padding = 1
        else:
            # vertical, height 
            if dir == 0:
                asy_kernel = (3, 1)
                padding = (1, 0)
                if self.opt.debug:
                    print("============== asymmetric block created ==============")
            # horizontal, width
            elif dir == 1:
                asy_kernel = (1, 3)
                padding = (0, 1)
                if self.opt.debug:
                    print("============== asymmetric block created ==============")
            else:
                asy_kernel = 3
                padding = 1
                if self.opt.debug:
                    print("============== symmetric block created ==============")

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=asy_kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * AsymmetricBottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * AsymmetricBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * AsymmetricBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * AsymmetricBottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * AsymmetricBottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x