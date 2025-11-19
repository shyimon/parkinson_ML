import numpy as np
import sys
np.set_printoptions(precision=3, suppress=True)

class neural_network:
    def __init__(self):
        self.network_structure = []
        
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
    
    def feed_forward(self, input_layer):
        feeding = np.array(input_layer)
        for hidden_layer in self.network_structure:
            hidden_layer = np.array(hidden_layer)
            feeding = feeding.dot(hidden_layer)
            feeding = self.__tanh_activation(feeding)
        return feeding
    
    def MSE_loss(self, output, Y, total_examples):
        s =(np.square(output-Y))
        s = np.sum(s)/total_examples
        return(s)
    
    def backprop(self, inputs, target, learning_rate=1):
        for layer in range(len(self.network_structure)):
            error = target - float(self.feed_forward(inputs))
            print(f"Predicted value is {self.feed_forward(inputs)}, while real value was {target}: error is {error}")


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
                output = self.feed_forward(input)
                epoch_losses.append(self.MSE_loss(output, Y[input], len(Y)))
                self.network_structure = self.backprop(input, Y[input])
            print(f"Accuracy for epoch {ep+1} = {(1-(sum(epoch_losses)/len(x))*100)}")
            accuracy.append(1-(sum(epoch_losses)/len(x))*100)
            loss.append(sum(epoch_losses) / len(x))
        return(self.network_structure, accuracy, loss)