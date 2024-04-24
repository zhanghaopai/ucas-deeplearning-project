import configparser

import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class ConvNet(Module):
    def __init__(self, in_channel, classes_num, device, config) -> None:
        super().__init__()
        self.device=device
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=config.getint("Conv", "out_channel1"),
                               kernel_size=config.getint("Conv", "kernal_size1"),
                               padding=config.getint("Conv", "padding"),
                               stride=config.getint("Conv", "stride"),
                               device=device)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=config.getint("Conv", "out_channel1"),
                               out_channels=config.getint("Conv", "out_channel2"),
                               kernel_size=config.getint("Conv", "kernal_size2"),
                               padding=config.getint("Conv", "padding"),
                               stride=config.getint("Conv", "stride"),
                               device=device)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=config.getint("Conv", "out_channel2"),
                               out_channels=config.getint("Conv", "out_channel3"),
                               kernel_size=config.getint("Conv", "kernal_size2"),
                               padding=config.getint("Conv", "padding"),
                               stride=config.getint("Conv", "stride"),
                               device=device)
        self.relu3 = nn.ReLU()

        self.pooling = nn.MaxPool2d(kernel_size=config.getint("MaxPooling", "kernal_size"),
                                    stride=config.getint("MaxPooling", "stride"))

        self.dropout1=nn.Dropout2d(config.getfloat("Conv", "dropout1"))

        image_width = int(32 / config.getint("MaxPooling", "kernal_size"))

        self.fc1 = nn.Linear(in_features=config.getint("Conv", "out_size"), out_features=128)
        self.dropout2=nn.Dropout(config.getfloat("Conv", "dropout2"))
        self.fc2 = nn.Linear(128, classes_num)

    def forward(self, x):
        x.to(self.device)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("../config.ini")

    conv = ConvNet(3, 10, "cpu", config)
    print(conv)
