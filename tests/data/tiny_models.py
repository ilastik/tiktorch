import logging

import torch

logger = logging.getLogger(__name__)


class TestModel0(torch.nn.Module):
    def __init__(self, N=7, D_in=3, H=4, D_out=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

        torch.manual_seed(0)
        self.x = torch.randn(N, D_in)
        self.y = self.forward(self.x).detach()

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class TinyConvNet2d(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 16, 3)
        self.nlin1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 64, 3)
        self.nlin2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(64, out_channels, 3)
        self.nlin3 = torch.nn.Sigmoid()

    def forward(self, x):
        return torch.nn.Sequential(self.conv1, self.nlin1, self.conv2, self.nlin2, self.conv3, self.nlin3)(x)


class TinyConvNet3d(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, 16, 3)
        self.nlin1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(16, 64, 3)
        self.nlin2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv3d(64, out_channels, 3)
        self.nlin3 = torch.nn.Sigmoid()

    def forward(self, x):
        return torch.nn.Sequential(self.conv1, self.nlin1, self.conv2, self.nlin2, self.conv3, self.nlin3)(x)


def train(model, x, y):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(10):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        logger.debug(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    test1 = TestModel0()
    print("train", test1.training)
    # test.to(dtype=torch.float)
    y = test1(test1.x)
    assert y.allclose(test1.y)

    test2 = test1.__class__()
    test2.eval()
    test2.load_state_dict(test1.state_dict())
    print("here train2", test2.training)
    print(id(test1.state_dict()))
    print(id(test2.state_dict()))

    sd = test2.state_dict()
    for key, value in sd.items():
        sd[key] = torch.zeros_like(value)

    print("train2", test2.training)

    print(id(test1) == id(test2))
    print(id(test1.state_dict()["linear1.weight"]) == id(test2.state_dict()["linear1.weight"]))
    print(id(test1.state_dict()), test1.state_dict()["linear1.weight"])
    print(id(test2.state_dict()), test2.state_dict()["linear1.weight"])
    test2.load_state_dict(sd)
    print(id(test1) == id(test2))
    magic1 = test1.state_dict()
    magic2 = test2.state_dict()
    print(id(test1.state_dict()) == id(test2.state_dict()))
    print(id(magic1) == id(magic2))
    # print(id(magic1['linear1.weight']) == id(magic2['linear1.weight']))
    # print(id(test1.state_dict()['linear1.weight']) == id(test2.state_dict()['linear1.weight']))
    print(id(test1.state_dict()), test1.state_dict()["linear1.weight"])
    print(id(test2.state_dict()), test2.state_dict()["linear1.weight"])

    print("train2", test2.training)
