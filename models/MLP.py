from torch.nn import Module, Linear, Dropout
import torch.nn.functional as F

import configparser


class MLP(Module):
    def __init__(self, input_size, classes_num, device, confi, active_function=F.relu):
        self.input_size=input_size
        self.classes_num=classes_num
        self.device=device
        self.active_function=active_function
        super().__init__()
        self.hidden_layer1 = Linear(input_size, config.getint("MLP","hidden1"), device=device)
        self.dropout= Dropout(0.5)
        self.hidden_layer2 = Linear(config.getint("MLP","hidden1"), config.getint("MLP","hidden2"), device=device)
        self.out_layer = Linear(config.getint("MLP","hidden2"), classes_num, device=device)



    def forward(self, x):
        x.to(self.device)
        x = x.reshape(-1, self.input_size)
        x = self.active_function(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.active_function(self.hidden_layer2(x))
        x = self.active_function(self.out_layer(x))
        result = F.log_softmax(x, dim=1)
        return result



if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read("../config.ini")

    mlp = MLP(3*32*32, 10, "cpu", config)
    print(mlp)

