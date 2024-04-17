import configparser

import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class ConvNet(Module):
    def __init__(self, in_channel, classes_num, device) -> None:
        super().__init__()
        config = configparser.ConfigParser()
        config.read("config.ini")

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=config.getint("Conv", "out_channel1"),
                               kernel_size=config.getint("Conv", "kernal_size"),
                               padding=config.getint("Conv", "padding"),
                               stride=config.getint("Conv", "stride"),
                               device=device)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=config.getint("MaxPooling", "kernal_size"),
                                    stride=config.getint("MaxPooling", "stride"))

        image_width = int(32 / config.getint("MaxPooling", "kernal_size"))
        self.image_size = config.getint("Conv", "out_channel1") * image_width * image_width
        self.fc1 = nn.Linear(in_features=self.image_size, out_features=128)
        self.fc2 = nn.Linear(128, classes_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
