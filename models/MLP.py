from torch.nn import Module
import torch
import torch.nn.functional as F


class MLP(Module):
    def __init__(self, in_size, out_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size=128
        self.hidden_layer1 = torch.nn.Linear(in_size, hidden_size)
        self.hidden_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out_layer = torch.nn.Linear(hidden_size, out_size)



    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.out_layer(x))
        return F.log_softmax(x)

