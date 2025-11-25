import neuron
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    # network_structure is the number of neurons in input, hidden layers
    # and output, expressed as an array. For example [6, 2, 2, 1].
    def __init__(self, network_structure, eta=0.1):
        self.eta = eta
        self.loss_history = {"training": [], "test": []}

        self.hidden_layers = []
        for i in range(len(network_structure) - 1):
            self.hidden_layers.append(
                                [neuron.Neuron(num_inputs=network_structure[0], 
                                index_in_layer=j, 
                                activation_function_type="tanh") for j in range(network_structure[i + 1])]
                                )