from torch.nn import Module


class ConvNet(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
