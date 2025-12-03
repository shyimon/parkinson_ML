import neuron
import numpy as np
import matplotlib.pyplot as plt
import utils

class NeuralNetwork:
    # Constructor
    # network_structure is the number of neurons in input, hidden layers
    # and output, expressed as an array. For example [6, 2, 2, 1].
    def __init__(self, network_structure, eta=0.1):
        self.eta = eta
        self.loss_history = {"training": [], "test": []}
        self.loss_type = "half_mse"

        self.layers = []
        self.layers.append([neuron.Neuron(num_inputs=0, index_in_layer=j, activation_function_type="sigmoid", is_output_neuron=False) for j in range(network_structure[0])])
        for i in range(len(network_structure) - 1):
            self.layers.append([neuron.Neuron(num_inputs=network_structure[i], index_in_layer=j, activation_function_type="sigmoid", is_output_neuron=(i==len(network_structure)-2)) for j in range(network_structure[i + 1])])

        for l in range(len(self.layers) - 1):
            for n in self.layers[l]:
                n.attach_to_output(self.layers[l + 1])

    # a single example is propagated through the network, calling
    # the feed_neuron method for each neuron
    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = [n.feed_neuron(x) for n in self.layers[l + 1]]
        return x

    # streamlines and encapsulates the forwarding of multiple examples
    def predict(self, X):
        preds = []
        for xi in X:
            out = self.forward(xi)
            if isinstance(out, (list, np.ndarray)):
                preds.append(float(out[0]))
            else:
                preds.append(float(out))
        return np.array(preds)

    # backprop implementation
    def backward(self, y_true):
        for l in reversed(self.layers[1:]):
            for n in l:
                n.compute_delta(y_true)

        for l in range(len(self.layers) - 1, 0, -1):
            for n in self.layers[l]:
                n.update_weights(self.eta)

    # core training method.
    # the test set is passed purely to assess the test error at each step but is not used for
    # learning, to keep the test set "unseen".
    # Nothing is returned because the network's weights are updated in place. (we choose to have a stateful network)
    def fit(self, X, X_test, y, y_test, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(epochs):
            total_loss = 0.0
            test_loss = 0.0
            for xi, yi in zip(X, y):
                outputs = self.forward(xi)
                y_pred = outputs[0]
                total_loss += self.compute_loss(yi, y_pred, loss_type=self.loss_type)

                self.backward(yi)

            avg_loss = total_loss / len(X)
            self.loss_history["training"].append(avg_loss)

            y_pred_test = self.predict(X_test)
            test_loss = np.sum(self.compute_loss(y_test, y_pred_test.flatten(), loss_type=self.loss_type))
            avg_test_loss = test_loss / len(y_test)
            self.loss_history["test"].append(avg_test_loss)
            if epoch % 25 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        utils.draw_network(self.layers)

    # a method to save the losses of the training and test sets as a plot
    def save_plots(self, path):
        plt.plot(self.loss_history["training"], label='Training Loss')
        plt.plot(self.loss_history["test"], label='Test Loss')
        plt.legend()
        plt.ylim(0, max(max(self.loss_history["training"]), max(self.loss_history["test"])) * 1.1)
        plt.savefig(path)
        
    def compute_loss(self, y_true, y_pred, loss_type="half_mse"):
        """
        Calcola la loss per un singolo esempio (quando verrà implementato il mini-batch
        adatterò questa funzione di conseguenza) e gestisce sia valori singoli che array.
        Args:
            y_true (float o array): Valore target vero.
            y_pred (float o array): Valore predetto dalla rete neurale.
            loss_type (str): Tipo di loss da calcolare.
        Returns:
            float o array: Valore della loss calcolata.
        """
        if loss_type == "half_mse":
            return 0.5 * (y_true - y_pred) ** 2
        else:
            raise ValueError(f"Loss type '{loss_type}' not implemented.")