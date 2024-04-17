from torch.nn import Module, Linear
import torch.nn.functional as F


class MLP(Module):
    def __init__(self, input_size, classes_num):
        self.input_size=input_size
        self.classes_num=classes_num
        super().__init__()
        hidden_size=128
        self.hidden_layer1 = Linear(input_size, hidden_size)
        self.hidden_layer2 = Linear(hidden_size, hidden_size)
        self.out_layer = Linear(hidden_size, classes_num)



    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.out_layer(x))
        result = F.log_softmax(x, dim=1)
        return result

