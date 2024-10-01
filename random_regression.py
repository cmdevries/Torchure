import torch.nn as nn
import torch
from torch.optim import SGD

def main():
    device = "cpu"
    hidden = 8
    model = nn.Sequential(
        nn.Linear(2, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)    
    loss = nn.MSELoss()
    opt = SGD(model.parameters(), lr=0.001)
    examples = 1000
    x = torch.rand(examples, 2)
    y = torch.rand(examples, 1)
    steps = 100
    for i in range(steps):
        for xx, yy in zip(x, y):
            opt.zero_grad()
            curr_loss = loss(model(xx), yy)
            curr_loss.backward()
            opt.step()
        print(f"step {i} = {curr_loss}")

if __name__ == "__main__":
    main()
