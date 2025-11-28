import numpy as np
import math

class Neuron:
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False, activation_function_type="tanh"):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron
        self.weights = np.random.uniform(-0.5, 0.5, size=num_inputs)
        self.bias = np.random.uniform(-0.5, 0.5) # la scelta del valore del bias è da giustificare con una grid search
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.inputs = []
        self.in_output_neurons = []
        self.activation_function_type = activation_function_type
    
    def attach_to_output(self, neurons):
        self.in_output_neurons = list(neurons)
    
    # Activation function gets called dynamically based on how the neuron was initialized
    def activation_funct(self, input):
        if self.activation_function_type == "sigmoid":
            return(1/(1 + np.exp(-input)))
        elif self.activation_function_type == "tanh":
            return((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)))
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
    
    # Activation function's derivative gets called dynamically based on how the neuron was initialized
    def activation_deriv(self, input):
        if self.activation_function_type == "sigmoid":
            return input * (1 - input)
        elif self.activation_function_type == "tanh":
            return 1 - np.square(input)
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")

    def feed_neuron(self, inputs):
        self.inputs = np.array(inputs, dtype=float)
        # print(f"\n\nInputs: {self.inputs}\nWeights:{self.weights}")
        self.net = float(np.dot(self.weights, inputs)) + self.bias
        self.output = self.activation_funct(self.net)
        return self.output

    def compute_delta(self, target):
        if self.is_output_neuron:
            self.delta = (target - self.output) * self.activation_deriv(self.output)
        else:
            delta_sum = 0.0
            for k in self.in_output_neurons:
                w_kj = k.weights[self.index_in_layer]
                delta_sum += k.delta * w_kj
                self.delta = delta_sum * self.activation_deriv(self.output)
        return self.delta

    def update_weights(self, eta):
        self.weights += eta * self.delta * self.inputs
        self.bias += eta * self.delta