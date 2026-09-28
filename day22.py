import numpy as np
import torch

y_true = 1

for p in [0.9, 0.5, 0.1]:
    mse = (p - y_true) ** 2
    ce = -np.log(p)
    print(f"prediction={p}: squared error={mse:.3f}, cross-entropy{ce:.3f}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y = 1

for z in [2.0, -2.0, -6.0]:
    p = sigmoid(z)
    graded_mse = 2 * (p - y) * p * (1 - p)
    graded_ce = p - y
    print(f"z={z}: p={p:.4f}, MSE Gradient={graded_mse:.4f}, CE Gradient={graded_ce:.4f}")

def run(name, lr, steps=100):
    w = torch.tensor([5.0, 5.0], requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr) if name == "SGD" else torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = w[0]**2 + 25 * w[1]**2
        loss.backward()
        opt.step()
    return w.detach().numpy().round(4), round(loss.item(), 4)

print("SGD lr=0.01:", run("SGD", 0.01))
print("Adam lr=0.1:", run("Adam", 0.1))
print("SGD lr=0.1:", run("SGD", 0.1))