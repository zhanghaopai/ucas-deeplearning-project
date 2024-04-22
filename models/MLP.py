from torch.nn import Module, Linear
import torch.nn.functional as F

import configparser


class MLP(Module):
    def __init__(self, input_size, classes_num, config):
        self.input_size=input_size
        self.classes_num=classes_num
        super().__init__()
        self.hidden_layer1 = Linear(input_size, config.getint("MLP","hidden1"))
        self.hidden_layer2 = Linear(config.getint("MLP","hidden1"), config.getint("MLP","hidden2"))
        self.out_layer = Linear(config.getint("MLP","hidden2"), classes_num)



    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.out_layer(x))
        result = F.log_softmax(x, dim=1)
        return result



if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read("../config.ini")

    mlp = MLP(3*32*32, 10, config)
    print(mlp)

