import neuron
import numpy as np
import matplotlib.pyplot as plt
import utils

class NeuralNetwork:
    def __init__(self, network_structure, num_inputs=6, num_hidden=4, num_outputs=1, eta=0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.eta = eta
        self.loss_history = {"training": [], "test": []}

        self.hidden_layer = [neuron.Neuron(num_inputs=num_inputs, index_in_layer=j, activation_function_type="tanh") for j in range(num_hidden)]
        self.output_layer = [neuron.Neuron(num_inputs=num_hidden, index_in_layer=k, is_output_neuron=True, activation_function_type="tanh") for k in range(num_outputs)]

        for h in self.hidden_layer:
            h.attach_to_output(self.output_layer)


    def forward(self, x):
        hidden_outputs = [h.feed_neuron(x) for h in self.hidden_layer]
        outputs = [o.feed_neuron(hidden_outputs) for o in self.output_layer]
        return hidden_outputs, outputs
    
    def predict(self, X):
        preds = []
        for xi in X:
            _, outputs = self.forward(xi)
            preds.append(outputs[0])
        return np.array(preds)
    
    def backward(self, x, y_true):
        if self.num_outputs == 1:
            y_true = [y_true]

        for k, o in enumerate(self.output_layer):
            o.compute_delta(y_true[k])

        for h in self.hidden_layer:
            h.compute_delta(y_true[k])

        hidden_outputs = [h.output for h in self.hidden_layer]
        for o in self.output_layer:
            o.inputs = np.array(hidden_outputs, dtype=float)
            o.update_weights(self.eta)

        for h in self.hidden_layer:
            h.update_weights(self.eta)
   
    def fit(self, X, X_test, y, y_test, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(epochs):
            total_loss = 0.0
            test_loss = 0.0
            for xi, yi in zip(X, y):
                _, outputs = self.forward(xi)
                y_pred = outputs[0]
                total_loss += 0.5 * (yi - y_pred) ** 2

                self.backward(xi, yi)

            avg_loss = total_loss / len(X)
            self.loss_history["training"].append(avg_loss)

            y_pred_test = self.predict(X_test)
            test_loss = 0.5 * sum(y_test - y_pred_test) ** 2
            avg_test_loss = test_loss / len(y_test)
            self.loss_history["test"].append(avg_test_loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def save_plots(self, path):
        plt.plot(self.loss_history["training"], label='Training Loss')
        plt.plot(self.loss_history["test"], label='Test Loss')
        plt.legend()
        plt.ylim(0, max(self.loss_history["training"]))
        plt.savefig(path)