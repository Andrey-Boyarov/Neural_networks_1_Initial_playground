from Layer_2_0 import Layer2

import numpy as np
import numpy.typing as npt


class Network2:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer2):
        self.layers.append(layer)
        return self

    def run(self, inputs: npt.NDArray):
        current = np.array(inputs)
        for layer in self.layers:
            current = np.array(layer.run(current))
        return current

    def get_layer(self, index: int):
        return self.layers[index]
