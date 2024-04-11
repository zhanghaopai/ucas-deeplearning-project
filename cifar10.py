import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 超参数定义
BATCH_SIZE = 512  # batch size
EPOCHS = 20  # 训练批次
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


train_feature, train_label = next(iter(train_dataloader))
img = train_feature[0].squeeze()
img = img.permute(1, 2, 0)
print(img.shape)
label = train_label[0]
plt.imshow(img)
plt.show()
