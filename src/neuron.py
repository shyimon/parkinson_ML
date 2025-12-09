import numpy as np
import math

class Neuron:
    # Constructor
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False, activation_function_type="tanh", weight_initializer='xavier', bias_initializer='uniform'):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron
        self.weights = np.random.uniform(-0.5, 0.5, size=num_inputs)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.activation_function_type = activation_function_type
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.inputs = []
        self.attached_neurons = []
        
        # Implementazione di una memoria per backprop e rprop
        self.prev_weight_grad = np.zeros(num_inputs) # Per ricordare il gradiente dei pesi al passo precedente (t-1)
        self.prev_bias_grad = 0.0 # Inizializza a zero la memoria del gradiente del bias al passo precedente 
        
        # Per quickprop
        self.prev_weight_update = np.zeros(num_inputs) # Per ricordare l'ultimo aggiornamento dei pesi (t-1)
        self.prev_bias_update = 0.0 # Inizializza a zero la memoria dell'ultimo aggiornamento del bias
        
        # Per rprop
        self.rprop_step_w = np.full(num_inputs, 0.1) # Serve un valore del passo specifico per ogni peso (dal momento che non usa un eta globale) che viene inizializzato a 0.1
        self.rprop_step_b = 0.1 # Lo stesso per il bias
        
    # Updates the list of output neurons
    def attach_to_output(self, neurons):
        self.attached_neurons = list(neurons)
    
    # Activation function gets called dynamically (string based) based on how the neuron was initialized
    def activation_funct(self, input):
        if self.activation_function_type == "sigmoid":
            return(1/(1 + np.exp(-input)))
        elif self.activation_function_type == "tanh":
            return((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)))
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
    
    # Activation function's derivative gets called dynamically (string based) based on how the neuron was initialized
    def activation_deriv(self, input):
        if self.activation_function_type == "sigmoid":
            return input * (1 - input)
        elif self.activation_function_type == "tanh":
            return 1 - np.square(input)
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")

    # a single example is fed to the neuron. The sum and then whatever activation function was selected are called
    def feed_neuron(self, inputs):
        self.inputs = np.array(inputs, dtype=float)
        self.net = float(np.dot(self.weights, inputs)) + self.bias
        self.output = self.activation_funct(self.net)
        return self.output
    
    # The delta computation for an output neuron and a hidden neuron are different, but
    # this method compounds them, checking the is_output_neuron flag (set at __init__ time)
    def compute_delta(self, signal_error): # target viene sostituito con signal_error che arriva direttamente da neural_network.backward
        if self.is_output_neuron:
            self.delta = signal_error * self.activation_deriv(self.output)
        else:
            delta_sum = 0.0
            for k in self.attached_neurons:
                w_kj = k.weights[self.index_in_layer]
                delta_sum += k.delta * w_kj
            self.delta = delta_sum * self.activation_deriv(self.output)
        return self.delta

    # Update of the weights based on delta and learning rate.
    # The updated weights are the "incoming" edges!
    def update_weights(self, eta):
        self.weights += eta * self.delta * self.inputs
        self.bias += eta * self.delta