from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import math

def normalize_data(X):
    """Normalizza i dati nell'intervallo [0, 1]"""
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

def return_monk1():
    train_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
    test_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    train_set = pd.read_csv(train_set_url, header=None, names=column_names, delim_whitespace=True)
    test_set = pd.read_csv(test_set_url, header=None, names=column_names, delim_whitespace=True)

    # Separazione corretta
    X_train = train_set.drop(columns=['class', 'id']).to_numpy()
    y_train = train_set['class'].to_numpy()
    
    X_test = test_set.drop(columns=['class', 'id']).to_numpy()
    y_test = test_set['class'].to_numpy()
    
    return X_train, y_train, X_test, y_test

class Neuron:
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron
        self.weights = np.random.uniform(-0.5, 0.5, size=num_inputs)
        self.bias = np.random.uniform(-0.5, 0.5) # la scelta del valore del bias è da giustificare con una grid search
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.inputs = None
        self.in_output_neurons = []

    def attach_to_output(self, neurons):
        self.in_output_neurons = list(neurons)

    def sigmoid(self, x):
        # Aggiunta di clipping per evitare overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + math.exp(-x))

    def derivative_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def predict(self, inputs):
        inputs = np.array(inputs, dtype=float)
        self.inputs = inputs
        self.net = float(np.dot(self.weights, inputs)) + self.bias
        self.output = self.sigmoid(self.net)
        return self.output

    def compute_delta_output(self, target):
        self.delta = (target - self.output) * self.derivative_sigmoid(self.net)
    
    def compute_delta_hidden(self):
        delta_sum = 0.0
        for k in self.in_output_neurons:
            w_kj = k.weights[self.index_in_layer]
            delta_sum += k.delta * w_kj
        self.delta = delta_sum * self.derivative_sigmoid(self.net)

    def update_weights(self, eta):
        self.weights += eta * self.delta * self.inputs
        self.bias += eta * self.delta

class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, num_outputs=1, eta=0.2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.eta = eta
        self.loss_history = []

        self.hidden_layer = [Neuron(num_inputs=num_inputs, index_in_layer=j) for j in range(num_hidden)]
        self.output_layer = [Neuron(num_inputs=num_hidden, index_in_layer=k, is_output_neuron=True) for k in range(num_outputs)]

        for h in self.hidden_layer:
            h.attach_to_output(self.output_layer)

    def forward(self, x):
        hidden_outputs = [h.predict(x) for h in self.hidden_layer]
        outputs = [o.predict(hidden_outputs) for o in self.output_layer]
        return hidden_outputs, outputs
    
    def backward(self, x, y_true):
        if self.num_outputs == 1:
            y_true = [y_true]

        for k, o in enumerate(self.output_layer):
            o.compute_delta_output(y_true[k])

        for h in self.hidden_layer:
            h.compute_delta_hidden()

        hidden_outputs = [h.output for h in self.hidden_layer]
        for o in self.output_layer:
            o.inputs = np.array(hidden_outputs, dtype=float)
            o.update_weights(self.eta)

        for h in self.hidden_layer:
            h.update_weights(self.eta)
   
    def fit(self, X, y, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                _, outputs = self.forward(xi)
                y_pred = outputs[0]
                total_loss += 0.5 * (yi - y_pred) ** 2
                self.backward(xi, yi)

            if epoch % 100 == 0:
                avg_loss = total_loss / len(X)
                self.loss_history.append(avg_loss)
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        preds = []
        for xi in X:
            _, outputs = self.forward(xi)
            preds.append(outputs[0])
        return np.array(preds)

# === MAIN EXECUTION ===
# Carica i dati
X_train, y_train, X_test, y_test = return_monk1()

# Normalizza i dati
X_train_normalized = normalize_data(X_train)
X_test_normalized = normalize_data(X_test)

# Parametri della rete
num_inputs = X_train_normalized.shape[1]
num_hidden = 4      # Ridotto per MONK-1
num_outputs = 1
eta = 0.2           # Learning rate aumentato

# Crea e allena la rete 
mlp = MultiLayerPerceptron(num_inputs, num_hidden, num_outputs, eta)
print("Inizio training...")
mlp.fit(X_train_normalized, y_train, epochs=500)

# Predizioni e accuracy
print("\nCalcolo accuracy...")
y_pred = mlp.predict(X_train_normalized)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)

accuracy = np.mean(y_pred_class == y_train) * 100
print(f"\nFinal Training Accuracy: {accuracy:.2f}%")

# Test accuracy
y_pred_test = mlp.predict(X_test_normalized)
y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
test_accuracy = np.mean(y_pred_test_class == y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Dettagli aggiuntivi
print(f"\nDettagli:")
print(f"Pattern corretti: {np.sum(y_pred_class == y_train)}/{len(y_train)}")
print(f"Pattern test corretti: {np.sum(y_pred_test_class == y_test)}/{len(y_test)}")