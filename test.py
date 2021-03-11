import torch

x = torch.randn(2, requires_grad = False)
y = x * 3
gradients = torch.tensor([100,0.1], dtype=torch.float)
y.backward(gradients)
print(x.grad)
