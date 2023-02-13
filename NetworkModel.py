from typing import Callable

import numpy as np
import numpy.typing as npt

from Layer import Layer


class NetworkModel:

    def __init__(self,
                 inputs: npt.NDArray,  # 1 dimension
                 weights_array: npt.NDArray,   # 3 dimensions
                 biases_array: npt.NDArray,    # 2 dimensions
                 f: Callable[[float], float],
                 last_f: Callable[[float], float]):

        if weights_array.shape[0] != biases_array.shape[0]:
            error_str = 'Weights array shape {we_ar_shape} is not equal to biases array shape {bi_ar_shape}'
            raise ValueError(error_str.format(we_ar_shape=weights_array.shape[0], bi_ar_shape=biases_array.shape[0]))

        self.__num_of_layers = weights_array.shape[0]

        self.__inputs = inputs
        self.__weights_array = weights_array
        self.__biases_array = biases_array
        self.__f = f
        self.__last_f = last_f

    def run(self) -> npt.NDArray:
        cur_inputs = self.__inputs

        for layer_index in range(self.__num_of_layers - 1):
            cur_inputs = self.__form_hidden_layer(cur_inputs, layer_index).run()

        return self.__form_last_layer(cur_inputs).run()

    def run_v1(self) -> npt.NDArray:
        cur_inputs = self.__inputs

        for layer_index in range(self.__num_of_layers - 1):
            cur_inputs = self.__form_hidden_layer(cur_inputs, layer_index).run_v1()

        return self.__form_last_layer(cur_inputs).run_v1()

    def __form_hidden_layer(self, inputs: np.array, layer_index: int) -> Layer:
        return Layer(
            inputs,
            self.__weights_array[layer_index],
            self.__biases_array[layer_index],
            self.__f)

    def __form_last_layer(self, inputs: npt.NDArray) -> Layer:
        return Layer(
            inputs,
            self.__weights_array[self.__num_of_layers - 1],
            self.__biases_array[self.__num_of_layers - 1],
            self.__last_f)
