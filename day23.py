# DROPOUT
import numpy as np
np.random.seed(0)

activations = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
p = 0.5
mask =np.random.rand(6) > p
print(mask)

dropped = activations * mask
print(dropped)

print(activations.sum())
print(dropped.sum())

dropped_scale = activations * mask / (1 - p)
print(dropped_scale)
print(dropped_scale.sum())

np.random.seed(1)
sums = []

for _ in range(10000):
    m = np.random.rand(6) > p
    trial = activations * m / (1 - p)
    sums.append(trial.sum())

print(np.mean(sums))
print("\n")

# BATCH NORMALIZATION
batch = np.array([50.0, 55.0, 45.0, 60.0])

mean = batch.mean()
std = batch.std()
print(mean, std)

epsilon = 1e-5
normalized = (batch - mean) / np.sqrt(std**2 + epsilon)
print(normalized)

print(normalized.mean())
print(normalized.std())

gamma = 2.0
beta = 1.0

output = gamma * normalized + beta
print(output)  

print(output.mean())
print(output.std())