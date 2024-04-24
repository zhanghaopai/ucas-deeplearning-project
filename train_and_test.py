import ssl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import time
import configparser
import matplotlib.pyplot as plt

from cifar.test import test
from cifar.train import train
from models.MLP import MLP
from models.ConvNet import ConvNet

# 下载cifar10数据集不使用ssl协议
ssl._create_default_https_context = ssl._create_unverified_context

# 超参数定义
config=configparser.ConfigParser()
config.read("config.ini")
BATCH_SIZE = config.getint("Training", "batch_size")
EPOCHS = config.getint("Training", "epochs")  # 训练批次
MOMENTUM = config.getfloat("Training", "momentum")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def batch(model, optimizer, learning_rate):
    print("此模型在", DEVICE, "上训练")  # 设备
    # 记录开始时间
    start_time = time.time()
    # 模型
    if(model == 'mlp'):
        real_model = MLP(input_size=input_size, classes_num=classes_num, device=DEVICE, config=config)
    elif (model == "cnn"):
        real_model= ConvNet(in_channel=3, classes_num=classes_num, device=DEVICE, config=config)
    # 优化器
    if (optimizer == "sgd"):
        real_optimizer = torch.optim.SGD(real_model.parameters(), lr=float(learning_rate), momentum=MOMENTUM)
    elif (optimizer == "adam"):
        real_optimizer = torch.optim.Adam(real_model.parameters(), lr=float(learning_rate))
    train_loss_list = []
    valid_loss_list = []
    accuracy_list=[]

    for epoch in range(EPOCHS):
        train_avg_loss = train(device=DEVICE,
                               train_loader=train_dataloader,
                               model=real_model,
                               optimizer=real_optimizer,
                               loss_function=F.cross_entropy)
        train_loss_list.append(train_avg_loss)

        vaild_avg_loss, valid_accuracy = test(device=DEVICE,
                                              test_loader=test_dataloader,
                                              model=real_model,
                                              loss_function=F.cross_entropy)
        valid_loss_list.append(vaild_avg_loss)
        accuracy_list.append(valid_accuracy)
        print("epoch: {}, train_loss: {}, test_loss: {}，accuracy:{}".format(epoch + 1, train_avg_loss, vaild_avg_loss,
                                                                            valid_accuracy))

    # 绘制loss曲线
    make_loss_plt(train_loss_list, valid_loss_list)
    # report
    predictions = []
    answers = []
    with torch.no_grad():
        for i, (image, label) in enumerate(test_dataloader):
            score = real_model(image.to(DEVICE))
            _, pred = torch.max(score, dim=1)
            predictions += list(pred.cpu().numpy())
            answers += list(label.cpu().numpy())
    print(classification_report(predictions, answers))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("训练+测试时长：",elapsed_time)


def make_loss_plt(train_loss_list, valid_loss_list, accuracy_list):
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(valid_loss_list, label="Testing Loss")
    plt.plot(accuracy_list, label="Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()