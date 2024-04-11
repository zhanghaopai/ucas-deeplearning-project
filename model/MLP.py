from torch.nn import Module


class MLP(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
