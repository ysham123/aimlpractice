import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

w = torch.tensor(1.0, requires_grad=True, device=device)

# First forward + backward
loss = (w - 4) ** 2
loss.backward()
print(w.grad)

# Gradient descent step
with torch.no_grad():
    w -= 0.1 * w.grad

# Clear gradients
w.grad.zero_()

# Second forward + backward
loss = (w - 4) ** 2
loss.backward()
print(w.grad)
