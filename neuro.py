from typing import List

import numpy as np
from numpy import ndarray





class Operation(object):
    '''
    Базовый класс операции в нейросети.
    '''
    def __init__(self):
pass
    def forward(self, input_: ndarray):
        '''
        Хранение ввода в атрибуте экземпляра self._input
        Вызов функции self._output().
        '''
        self.input_ = input_
        self.output = self._output()
        return self.output
    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Вызов функции self._input_grad().
        Проверка совпадения размерностей.
        '''
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad
    def _output(self) -> ndarray:
        '''
        Метод _output определяется для каждой операции.
        '''
        raise NotImplementedError()


    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Метод _input_grad определяется для каждой операции.
        '''
        raise NotImplementedError()


class NeuralNetwork(object):
    '''
    Класс нейронной сети.
    '''

    def __init__(self, layers: List[Layer],
                 loss: Loss,
                 seed: float = 1):
        '''
        Нейросети нужны слои и потери.
        '''



    self.layers = layers
    self.loss = loss
    self.seed = seed
    if seed:
        for layer in self.layers:
            setattr(layer, "seed", self.seed)


def forward(self, x_batch: ndarray) -> ndarray:
    '''
    Передача данных через последовательность слоев.
    '''
    x_out = x_batch
    for layer in self.layers:
        x_out = layer.forward(x_out)
    return x_out


def backward(self, loss_grad: ndarray) -> None:
    '''
    Передача данных назад через последовательность слоев.
    '''
    grad = loss_grad
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
    return None


def train_batch(self,
                x_batch: ndarray,
                y_batch: ndarray) -> float:
    '''
    Передача данных вперед через последовательность слоев.
    Вычисление потерь.
    Передача данных назад через последовательность слоев.
    '''
    predictions = self.forward(x_batch)
    loss = self.loss.forward(predictions, y_batch)
    self.backward(self.loss.backward())


    return loss


def params(self):
    '''
    Получение параметров нейросети.
    for layer in self.layers:
        yield from layer.params
def param_grads(self):
    '''

""""""
Получение
градиента
потерь
по
отношению
к
параметрам
нейросети.
""""""
for layer in self.layers:
            yield from layer.param_grads
