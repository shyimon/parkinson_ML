from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import math
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
x = monk_s_problems.data.features 
y = monk_s_problems.data.targets 

# metadata 
print(monk_s_problems.metadata) 
  
# variable information 
print(monk_s_problems.variables) 

def return_monk1():
    train_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
    test_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    train_set = pd.read_csv(train_set_url, header=None, names=column_names, delim_whitespace=True)
    test_set = pd.read_csv(test_set_url, header=None, names=column_names, delim_whitespace=True)

    return train_set, test_set

train_set, test_set = return_monk1()

print("Il tipo di variabili dei train e del test sets sono:", type(train_set), type(test_set))
print("Il tipo di variabile y è:", type(y))
train_set = train_set.drop(columns=["id"])
test_set = test_set.drop(columns=["id"])
train_set = train_set.to_numpy()
y: pd.Series
y = y.to_numpy().ravel()
y_train = y[:len(train_set)]

print(train_set)

class Neuron:
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron

        self.weights = np.random.uniform(0.01, 0.1, size=num_inputs)
        self.net = 0.0          # net input (scalare)
        self.output = 0.0       # output del neurone
        self.delta = 0.0
        self.epochs = 0.0
        self.inputs = None

        self.in_output_neurons = [] 

    def attach_to_output(self, neurons):
        self.in_output_neurons = list(neurons)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def predict(self, inputs):
        inputs = np.array(inputs, dtype=float)
        self.inputs = inputs             
        self.net = float(np.dot(self.weights, inputs))
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
        if self.inputs is None:
            raise ValueError("self.inputs è None: devi chiamare predict() prima di update_weights().")
        self.weights = self.weights + eta * self.delta * self.inputs

        self.weights = self.weights + eta * self.delta * self.inputs

class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, num_outputs=1, eta=0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.eta = eta

        # hidden layer
        self.hidden_layer = [Neuron(num_inputs=num_inputs, index_in_layer=j, is_output_neuron=False) for j in range(num_hidden)]

        # output layer
        self.output_layer = [Neuron(num_inputs=num_hidden, index_in_layer=k, is_output_neuron=True) for k in range(num_outputs)]

        # Connect each hidden neuron to the output neuron
        for h in self.hidden_layer:
            h.attach_to_output(self.output_layer)

    #forward pass
    def forward(self, x):
        hidden_outputs = [h.predict(x) for h in self.hidden_layer]
        outputs = [o.predict(hidden_outputs) for o in self.output_layer]
        return np.array(hidden_outputs, dtype=float), np.array(outputs, dtype=float)
    
    #backpropagation
    def backward(self, x, y_true):
        if self.num_outputs == 1:
            y_true = [y_true]

    # delta output neuron
        for k, o in enumerate(self.output_layer):
         o.compute_delta_output(y_true[k])

    # delta hidden neuron
        for h in self.hidden_layer:
            h.compute_delta_hidden()

    # 3) update output weights
        hidden_outputs = [h.output for h in self.hidden_layer]
        for o in self.output_layer:
            o.inputs = np.array(hidden_outputs, dtype=float)   # <--- IMPORTANTE
            o.update_weights(self.eta)

    # update hidden weights
        for h in self.hidden_layer:
            h.update_weights(self.eta)

   
    #training
    def fit(self, X, y, epochs=390):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(epochs):
            total_loss = 0.0

        for xi, yi in zip(X, y):
            # forward
            _, outputs = self.forward(xi)          
            y_pred = outputs[0] if self.num_outputs == 1 else outputs

            # loss
            total_loss += 0.5 * np.mean((y_pred - yi) ** 2)

            # backward + update
            self.backward(xi, yi)

        print(f"Epochs: {epochs} and loss: {total_loss/len(X):.4f}")


    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = []
        for xi in X:
         _, outputs = self.forward(xi)
        y_pred = outputs[0] if self.num_outputs == 1 else outputs
        preds.append(y_pred)
        return np.array(preds)
    
num_inputs = train_set.shape[1]
num_hidden = 3      
num_outputs = 1
eta = 0.1

mlp = MultiLayerPerceptron(num_inputs, num_hidden, num_outputs, eta)

mlp.fit(train_set, y_train, epochs=390)

y_pred = mlp.predict(train_set)
y_pred_class = 1 if y_pred >= 5 else 0

accuracy = np.mean(y_pred_class == y_train)
print("Accuracy:", accuracy * 100 "%")


