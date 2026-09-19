import numpy as np


# Input data
X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 2.2],
    [3.5, 4.0],
    [4.0, 4.5],
    [4.5, 4.2]
])

np.random.seed(42)

# Weights for the hidden layer
W1 = np.random.randn(2, 4)

# Bias for the hidden layer
b1 = np.zeros(4)


# Linear transformation
Z1 = X @ W1 + b1

    
# print("X shape:", X.shape)
# print("W1 shape:", W1.shape)
# print("b1 shape:", b1.shape)
# print("Z1 shape:", Z1.shape)

# print(Z1)

def relu(x):
    return np.maximum(0,x)

A1 = relu(Z1)

# print("Z1:")
# print(Z1)

# print("\nA1:")
# print(A1)

W2 = np.random.randn(4, 1)
b2 = np.zeros(1)

Z2 = A1 @ W2 + b2

# print("W2 shape:", W2.shape)
# print("b2 shape:", b2.shape)
# print("Z2 shape:", Z2.shape)
# print("\nZ2:")
# print(Z2)

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


A2 = sigmoid(Z2)

print(A2)
