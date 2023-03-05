from typing import Callable

import numpy as np
import numpy.typing as npt


class Layer2:

    def __init__(self, weights: npt.NDArray, biases: npt.NDArray, f: Callable[[float], float]):

        if biases.shape[0] != weights.shape[0]:
            error_str = 'Biases shape {in_shape} is not equal to weight shape(0) {bi_shape}'
            raise ValueError(error_str.format(in_shape=biases.shape[0], bi_shape=weights.shape[0]))

        self.__weights = weights  # 2 dimensions (1: biases, 2: inputs)
        self.__biases = biases    # 1 dimension
        self.__activation_f = f

        self.__size = weights.shape[0]  # size of this layer
        self.__prev_size = weights.shape[1]     # size of previous layer

    def run(self, inputs: npt.NDArray) -> np.array:
        nodes = np.zeros(self.__size)    # nodes of this layer

        for node_i in range(self.__size):
            nodes[node_i] = self.__calc_node_v1(node_i, inputs)

        return nodes

    def increase_bias(self, index: int, value: float):
        self.__biases[index] += value

    def increase_weights(self, indexes, value: float):
        self.__weights[indexes[1]][indexes[2]] += value

    def __calc_node_v1(self, node_i: int, inputs: npt.NDArray):
        node_value = self.__calc_all_links(node_i, inputs)
        return self.__activation_f(node_value)

    def __calc_all_links(self, node_i: int, inputs: npt.NDArray) -> float:
        synapses = np.dot(inputs, self.__weights[node_i]) + self.__biases[node_i]
        return float(np.sum(synapses))
