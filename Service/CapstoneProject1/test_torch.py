import torch
import torch.nn as nn

print(torch.__version__)

model = nn.Linear(10, 2)

x = torch.randn(1, 10)

print("before")

with torch.no_grad():
    y = model(x)

print("after")
print(y)