import ssl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import time
import configparser

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
LEARNING_RATE = config.getfloat("Training", "learning_rate") # 学习率
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

if __name__ == '__main__':
    print("此模型在", DEVICE, "上训练") # 设备
    # 记录开始时间
    start_time=time.time()
    # 模型
    # model = MLP(input_size=input_size, classes_num=classes_num)
    model = ConvNet(in_channel=3, classes_num=classes_num, device=DEVICE)
    # 优化器
    SGD_optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # adam_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss_list=[]
    valid_loss_list=[]

    for epoch in range(EPOCHS):
        train_avg_loss = train(device=DEVICE, train_loader=train_dataloader, input_size=input_size, model=model,
                               optimizer=SGD_optimizer, loss_function=F.cross_entropy)
        vaild_avg_loss, valid_accuracy = test(device=DEVICE, test_loader=test_dataloader, input_size=input_size, model=model,
                              loss_function=F.cross_entropy)
        print("epoch: {}, train_loss: {}, test_loss: {}，accuracy:{}".format(epoch + 1, train_avg_loss, vaild_avg_loss, valid_accuracy))
    # report
    predictions = []
    answers = []
    with torch.no_grad():
        for i, (image, label) in enumerate(test_dataloader):
            score = model(image)
            _, pred = torch.max(score, dim=1)
            predictions += list(pred.cpu().numpy())
            answers += list(label.cpu().numpy())
    print(classification_report(predictions, answers))
    # 记录结束时间
    end_time = time.time()
    elapsed_time=end_time - start_time
    print(elapsed_time)
