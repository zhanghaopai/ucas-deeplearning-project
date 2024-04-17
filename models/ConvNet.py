from torch.nn import Module
from torch import nn
import configparser
import torch.nn.functional as F


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

        self.output_layer = nn.Linear(in_features=config.getint("Conv", "out_channel1"), out_features=classes_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)
