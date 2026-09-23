import numpy as np

# a single perceptron: 3 inputs, 1 output
weights = np.array([0.5, -0.3, 0.8])
bias = 0.1

inputs = np.array([1.0, 2.0, 3.0])

weighted_sum = np.dot(inputs, weights) + bias
print(weighted_sum)

def relu(z):
    return max(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("ReLU:", relu(weighted_sum))
print("Sigmoid:", sigmoid(weighted_sum))

# Input: 3 features (same as before)
inputs = np.array([1.0, 2.0, 3.0])

# Hidden layer: 2 neurons, each with its own weights + bias for the 3 inputs
W1 = np.array([
    [0.5, -0.3, 0.8],    # neuron 1's weights
    [0.2, 0.4, -0.5],    # neuron 2's weights
])
b1 = np.array([0.1, -0.2])

hidden_z = W1 @ inputs + b1
hidden_output = np.maximum(0, hidden_z)    # ReLU applied to BOTH neurons at once
print("Hidden Layout Output:", hidden_output)

W2 = np.array([0.6, -0.9])      # output neuron's weights (2 inputs now, since hidden layer has 2 neurons)
b2 = 0.05

output_z = np.dot(hidden_output, W2) + b2
final_output = sigmoid(output_z)
print("Final Output:", final_output)