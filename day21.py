import numpy as np

sequence = [1.0, 2.0, 0.5]    # a "sequence" of 3 simple inputs (imagine 3 words, simplified to numbers)

w_input = 0.5
w_hidden = 0.8
bias = 0.1

hidden_state = 0.0    # starts at zero - no memory yet

for t, x in enumerate(sequence):
    hidden_state = np.tanh(w_input * x + w_hidden * hidden_state + bias)
    print(f"step {t}: input={x}, new hidden_state={hidden_state:.4f}")