import torch 
import torch.nn as nn

x = torch.tensor([[2.0, 3.0]])
model = nn.Linear(2,1)

y_hat = model(x)

print(y_hat)