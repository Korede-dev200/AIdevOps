import numpy as np

# a tiny 5x5 "image" - imagine this is a simplified grayscale picture
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0], 
])

# a 3x3 filter - this particular one is a classic "detect vertical edges" filter
vertical_edge_filter = ([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

print(image)
print(vertical_edge_filter)

def convolve_step(image_patch, filt):
    return np.sum(image_patch * filt)

# slide the 3x3 filter across every valid position in the 5x5 image
output = np.zeros((3, 3))    # 5x5 image, 3x3 filter -> 3x3 output

for i in range(3):
    for j in range(3):
        patch = image[i:i+3, j:j+3]
        output[i, j] = convolve_step(patch, vertical_edge_filter)

print(output)

def max_pool(x, size=2):
    h, w = x.shape
    out_h, out_w = h // size, w // size
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = x[i*size:i*size+size, j*size:j*size+size]
            output[i, j] = np.max(patch)
    return output

conv_output = np.array([[-2., 0., 2.], [-3., 0., 3.], [-2., 0., 2.]])
pooled = max_pool(conv_output, size=2)
print("Before Pooling:\n", conv_output)
print("After Pooling:\n", pooled)