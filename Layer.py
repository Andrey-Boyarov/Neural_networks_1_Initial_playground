from typing import Callable

import numpy as np
import numpy.typing as npt


class Layer:

    def __init__(self, inputs: npt.NDArray, weights: npt.NDArray, biases: npt.NDArray, f: Callable[[float], float]):

        if inputs.shape[0] != weights.shape[1]:
            error_str = 'Inputs shape {in_shape} is not equal to weight shape(1) {we_shape}'
            raise ValueError(error_str.format(in_shape=inputs.shape[0], we_shape=weights.shape[1]))
        if biases.shape[0] != weights.shape[0]:
            error_str = 'Biases shape {in_shape} is not equal to weight shape(0) {bi_shape}'
            raise ValueError(error_str.format(in_shape=biases.shape[0], bi_shape=weights.shape[0]))

        self.__inputs = inputs    # 1 dimension
        self.__weights = weights  # 2 dimensions (1: biases, 2: inputs)
        self.__biases = biases    # 1 dimension
        self.__activation_f = f

        self.__size = weights.shape[0]  # size of this layer
        self.__prev_size = weights.shape[1]     # size of previous layer

    # Returns output values for this layer
    def run(self) -> np.array:
        nodes = np.zeros(self.__size)    # nodes of this layer

        for node_i in range(self.__size):
            nodes[node_i] = self.__calc_node(node_i)

        return nodes

    def run_v1(self) -> np.array:
        nodes = np.zeros(self.__size)    # nodes of this layer

        for node_i in range(self.__size):
            nodes[node_i] = self.__calc_node_v1(node_i)

        return nodes

    # Returns value of specific node
    def __calc_node(self, node_i: int) -> float:
        node_value = 0.0

        for prev_node_i in range(self.__prev_size):
            node_value += self.__calc_link(node_i, prev_node_i)

        node_value += self.__biases[node_i]

        return self.__activation_f(node_value)

    def __calc_node_v1(self, node_i: int):
        node_value = self.__calc_all_links(node_i)

        return self.__activation_f(node_value)

    # Returns the result of specific node connection to another specific node
    def __calc_link(self, node_i: int, prev_node_i: int) -> float:
        return self.__inputs[prev_node_i] * self.__weights[node_i, prev_node_i]

    def __calc_all_links(self, node_i: int) -> float:
        synapses = np.dot(self.__inputs, self.__weights[node_i]) + self.__biases[node_i]
        return float(np.sum(synapses))
