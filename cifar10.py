import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.MLP import MLP
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# 超参数定义
BATCH_SIZE = 512  # batch size
EPOCHS = 20  # 训练批次
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARNING_RATE=0.01
MOMENTUM=0.9


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

input_size=3*32*32 # 输入大小
classes_num=10  # 分类数量
# 定义模型、优化器、loss
# model = MLP(input_size=input_size, output_size=output_size)
# optimizer=torch.optim.SGD(model.parameters(), LEARNING_RATE, MOMENTUM)


if __name__=='__main__':
    model = MLP(input_size=input_size, classes_num=classes_num)
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, MOMENTUM)



