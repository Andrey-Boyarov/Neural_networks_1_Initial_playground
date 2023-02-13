import numpy as np

from Layer import Layer
from NetworkModel import NetworkModel

inputs = np.array([0.8, 0.5])
sigmoid = lambda x: 1 / (1 + np.exp(-x))

biases1 = np.array([1, 2, 3])
weights1 = np.array([[0.1, 0.8], [0.4, 0.2], [0.7, 1]])

a = Layer(inputs, weights1, biases1, sigmoid)

biases2 = np.array([2, 0.4])
weights2 = np.array(weights1.T)
b = Layer(a.run(), weights2, biases2, sigmoid)

print(b.run())


model = NetworkModel(
    inputs,
    np.array([weights1, weights2], dtype=object),
    np.array([biases1, biases2], dtype=object),
    sigmoid,
    sigmoid)

print(model.run())
print(model.run_v1())
