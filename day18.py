import numpy as np
# Goal: learn a weight w such that w * x ≈ y, for known x and y
x = 2.0 
y_true = 10.0    # the "correct" answer we want the model to learn to produce

w = 0.0   # start with a bad guess

def predict(w, x):
    return w * x

def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2   # squared error

y_pred = predict(w, x)
current_loss = loss(y_pred, y_true)
print(f"w={w}, prediction={y_pred}, loss={current_loss}")

learning_rate = 0.01

gradient = 2 * (y_pred - y_true) * x
print (f"gradient: {gradient}")

w = w - learning_rate * gradient
print(f"updated_w: {w}")

w = 0.0   # reset
learning_rate = 0.01

for step in range(50):
    y_pred = predict(w, x)
    current_loss = loss(y_pred, y_true)
    gradient = 2 * (y_pred - y_true) * x
    w = w - learning_rate * gradient

    if step % 10 == 0:    # print every 10th step, not all 50
        print(f"step: {step}, w={w:.4f}, loss={current_loss:.4f}")

print(f"\nFinal w: {w:.4f} (target: w=5.0, since 5.0 * 2.0 = 10.0)")