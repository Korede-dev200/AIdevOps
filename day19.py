import torch

x = torch.tensor(2.0)
y_true = torch.tensor(10.0)

w = torch.tensor(0.0, requires_grad=True)  # <-- this is new

y_pred = w * x
loss = (y_pred - y_true) ** 2

print(loss)

loss.backward()
print(w.grad)

learning_rate = 0.01

with torch.no_grad():    # temporarily stop tracking, since this update itself shouldn't be part of the gradient chain
    w -= learning_rate * w.grad
    w.grad.zero_()    # clear the gradient before the next round

print(w)

w = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.01

for step in range(50):
    y_pred = w * x
    loss = (y_pred - y_true) ** 2

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()

    if step % 10 == 0:
        print(f"step {step}: w={w.item():.4f}, loss={loss.item():.4f}")

print(f"\nFinal w: {w.item():.4f}")

x = torch.tensor(2.0)
y_true = torch.tensor(10.0)

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)  # new: a second learnable parameter

learning_rate = 0.01

for step in range(50):
    y_pred = w * x + b
    loss = (y_pred - y_true) ** 2

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if step % 10 == 0:
        print(f"step {step}: w={w.item():.4f}, b={b.item():.4f}, loss={loss.item():.4f}")

print(f"\nFinal: w={w.item():.4f}, b={b.item():.4f}")