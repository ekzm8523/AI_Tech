import torch

x = torch.ones(2,2, requires_grad = True)
y = x + 2

z = y*y*3

out = z.mean()
print(out)
out.backward()	# out.backward(torch.tensor(1.)) 과 동일
# print(x.grad)
# print(x)
# print(y)
# print(z)


