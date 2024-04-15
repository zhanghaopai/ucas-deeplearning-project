import ssl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cifar.test import test
from cifar.train import train
from models.MLP import MLP

ssl._create_default_https_context = ssl._create_unverified_context

# 超参数定义
BATCH_SIZE = 512  # batch size
EPOCHS = 20  # 训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-3
MOMENTUM = 0.9

# 定义训练集和测试集
train_datasets = datasets.CIFAR10(
    root="data",
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    train=True,
    download=True,
)
test_datasets = datasets.CIFAR10(
    root="data",
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    train=False,
    download=True,
)
# 定义dataloader
train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False)

input_size = 3 * 32 * 32  # 输入大小
classes_num = 10  # 分类数量

if __name__ == '__main__':
    # 模型
    model = MLP(input_size=input_size, classes_num=classes_num)
    # 优化器
    # SGD_optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        train_avg_loss = train(device=DEVICE, train_loader=train_dataloader, input_size=input_size, model=model,
                               optimizer=adam_optimizer, loss_function=F.cross_entropy)
        vaild_avg_loss = test(device=DEVICE, test_loader=test_dataloader, input_size=input_size, model=model,
                              loss_function=F.cross_entropy)
        print("epoch: {}, train_loss: {}, test_loss: {}".format(epoch + 1, train_avg_loss, vaild_avg_loss))
