import numpy as np
import sys
from tqdm import tqdm
np.set_printoptions(precision=3, suppress=True)

class neural_network:
    def __init__(self):
        self.network_structure = []
        self.activations = []
        
    def add_layer(self, weights):
        if self.network_structure and len(self.network_structure[-1][1]) != len(weights):
            raise ValueError("Network structure must be coherent")
        self.network_structure.append(weights)

    def print_network_structure(self):
        print(self.network_structure)

    def __sigmoid_activation(self, input):
        return(1/(1 + np.exp(-input)))
    
    def __tanh_activation(self, input):
        return((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)))
    
    def __tanh_deriv(self, input):
        return 1 - np.square(input)
    
    def feed_forward(self, input_layer):
        self.activations = [np.array(input_layer).reshape(1, -1)]
        print([np.array(input_layer).reshape(1, -1)])
        feeding = np.array(input_layer)
        for hidden_layer in self.network_structure:
            hidden_layer = np.array(hidden_layer)
            feeding = feeding.dot(hidden_layer)
            feeding = self.__tanh_activation(feeding)
            feeding = np.array(feeding.reshape(1, -1))
            self.activations.append(feeding)
        return feeding
    
    def MSE_loss(self, output, Y, total_examples):
        s =(np.square(output-Y))
        s = np.sum(s)/total_examples
        return(s)
    
    def accuracy(self, predicts, Y):
        tot = len(Y)
        acc = 0
        for i in range(Y):
            if Y[i] == predicts[i]:
                acc += 1
        return acc / tot

    def backprop(self, target, learning_rate=0.05):
        target = np.array(target).reshape(1, -1)
        error = target - float(self.activations[-1])

        for layer in range(len(self.network_structure), 0, -1):
            delta = error * self.__tanh_deriv(self.activations[layer])
            previous_layer_error = np.dot(delta, self.network_structure[layer - 1].T)
            
            gradient = np.dot(self.activations[layer - 1].T, delta)
            # print(self.network_structure)
            self.network_structure[layer - 1] += learning_rate * gradient
            # print(self.network_structure)
            error = previous_layer_error
    

    # The training loop. It takes in input:
    # - x, the inputs
    # - Y, the targets
    # - net_structure, which is a list of matrices of weights. The sizes of matrices define the
    #   number of neurons and the values define the weights.
    # - hyperparameters, yet to define
    def train(self, x, Y, epochs=10):
        accuracy = []
        loss = []
        for ep in range(epochs):
            epoch_losses = []
            for input in range(len(x)):
                output = self.feed_forward(x.iloc[input])
                if ep == epochs - 1:
                    print(f"Input is {x.iloc[input]}")
                epoch_losses.append(self.MSE_loss(output, Y[input], len(Y)))
                self.backprop(Y[input], 0.2)
            # print(f"Accuracy for epoch {ep+1} = {(1-(sum(epoch_losses)/len(x))*100)}")
            # print(f"Loss for epoch {ep+1} = {sum(epoch_losses) / len(x)}\n\n")
            accuracy.append(1-(sum(epoch_losses)/len(x))*100)
            loss.append(sum(epoch_losses) / len(x))
        return(self.network_structure, accuracy, loss)