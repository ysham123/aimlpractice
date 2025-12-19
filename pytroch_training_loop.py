import torch 
import torch.nn as nn 

model = nn.Linear(1,1)   

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  
x = torch.tensor([[1.0],[2.0],[3.0]])
y = torch.tensor([[2.0],[4.0],[6.0]])

for epoch in range(1000):
    optimizer.zero_grad()         
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')
