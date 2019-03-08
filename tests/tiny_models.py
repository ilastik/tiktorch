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
        self.y = torch.randn(N, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


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
