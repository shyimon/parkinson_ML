import neuron
import numpy as np
import matplotlib.pyplot as plt
import utils

class NeuralNetwork:
    # network_structure is the number of neurons in input, hidden layers
    # and output, expressed as an array. For example [6, 2, 2, 1].
    def __init__(self, network_structure, eta=0.1):
        self.eta = eta
        self.loss_history = {"training": [], "test": []}

        self.layers = []
        self.layers.append([neuron.Neuron(num_inputs=0, index_in_layer=j, activation_function_type="tanh", is_output_neuron=False) for j in range(network_structure[0])])
        for i in range(len(network_structure) - 1):
            self.layers.append([neuron.Neuron(num_inputs=network_structure[i], index_in_layer=j, activation_function_type="tanh", is_output_neuron=(i==len(network_structure)-1)) for j in range(network_structure[i + 1])])

        for l in range(len(self.layers) - 1):
            for n in self.layers[l]:
                n.attach_to_output(self.layers[l + 1])

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = [n.feed_neuron(x) for n in self.layers[l + 1]]
        return x

    def predict(self, X):
        preds = []
        for xi in X:
            preds.append(self.forward(xi))
        return np.array(preds)

    def backward(self, y_true):
        for l in reversed(self.layers):
            for n in l:
                n.compute_delta(y_true)

        for l in range(len(self.layers) - 1, 0, -1):
            for n in self.layers[l]:
                n.update_weights(self.eta)


    def fit(self, X, X_test, y, y_test, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(epochs):
            total_loss = 0.0
            test_loss = 0.0
            for xi, yi in zip(X, y):
                outputs = self.forward(xi)
                y_pred = outputs[0]
                total_loss += 0.5 * (yi - y_pred) ** 2

                self.backward(yi)

            avg_loss = total_loss / len(X)
            self.loss_history["training"].append(avg_loss)

            y_pred_test = self.predict(X_test)
            test_loss = 0.5 * sum(y_test - y_pred_test) ** 2
            avg_test_loss = test_loss / len(y_test)
            self.loss_history["test"].append(avg_test_loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        utils.draw_network(self.layers)

    def save_plots(self, path):
        plt.plot(self.loss_history["training"], label='Training Loss')
        plt.plot(self.loss_history["test"], label='Test Loss')
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(path)