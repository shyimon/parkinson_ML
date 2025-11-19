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

print("Dimensione del train set:", len(train_set))
print("Dimensione del test set set:", len(test_set))
print("Il tipo di variabili dei train e del test sets sono:", type(train_set), type(test_set))
print("Il tipo di variabile y è:", type(y))
train_set = train_set.drop(columns=["id"])
test_set = test_set.drop(columns=["id"])
train_set = train_set.astype(float).to_numpy()
y = np.asarray(y, dtype=float).ravel()

print(train_set)

class Neuron():
    def __init__(self, position_in_layer, is_output_neuron=False):
        self.weights=[]
        self.inputs=[]
        self.output=None
        self.update_weights=[]
        self.is_output_neuron = is_output_neuron
        self.delta=None
        self.position_in_layer=position_in_layer
    
    def attach_to_output(self, neurons):
        self.in_output_neurons=[]

        for neuron in neurons:
            self.in_output_neurons.append(neuron)
    
    def sigmoid(self, x):
        return 1 / (1+math.exp(-x))
    
    def init_weights(self, num_input):
        for i in range(num_input):
         self.weights.append(np.random.uniform(0.01, 0.1))
    
    def predict(self, row):
        self.inputs=[]
        net = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            net += weight * feature
            self.output = self.sigmoid(net)
            return self.output
        
    def update_neuron(self):
        self.weights=[]
        for new_weight in self.update_weights:
            self.weights.append(new_weight)
    
    def calculate_update(self, eta, target):
        if self.is_output_neuron:
            self.delta=(self.output_target) * self.output * (1-self.output)
        else:
            delta_sum = 0
            cur_weight_index = self.position_in_layer
            for output_neuron in self.output_neuron:
                delta_sum += output_neuron.delta * output_neuron.weights[cur_weight_index]
                self.delta = delta_sum * self.output * self.output * (1-self.output)
                self.update_weights=[]

                for cur_weight, cur_input in zip(self.weights, self.inputs):
                    gradient = self.delta * cur_input
                    new_weight = cur_weight - eta * gradient
                    self.update_weights.append(new_weight)

class MultiLayerPerceptron:
    def __init__(self, num_neuron, eta, epochs):
        # one output neuron
        self.output_neuron = Neuron(0, is_output_neuron=True)

        self.perceptrons = []
        for i in range(num_neuron):
            # create neuron
            neuron = Neuron(i)
            # attach the output layer to this neuron
            neuron.attach_to_output([self.output_neuron])
            # append to layer
            self.perceptrons.append(neuron)

        # training parameters
        self.eta = eta
        self.epochs = epochs
        self.num_neuron = num_neuron
    #forward pass
    def predict(self, row):
        # row arriva come numpy.ndarray -> lo converto in lista
        row_list = list(row)          # NON modifichiamo 'row' originale
        row_list.append(1.0)          # bias per lo strato nascosto

        # attivazioni dei neuroni nascosti
        hidden_activations = [p.predict(row_list) for p in self.perceptrons]
        hidden_activations.append(1.0)     # bias per il neurone di output

        # output della rete
        net = self.output_neuron.predict(hidden_activations)

        # soglia a 0.5
        if net >= 0.5:
            return 1.0
        return 0.0
    
    def fit(self, train_set, y):
        # stochastic gradient descent
        num_row = len(train_set)
        num_feature = len(train_set[0])

    # Initialization of the weights
        for neuron in self.perceptrons:
            neuron.init_weights(num_feature)
        self.output_neuron.init_weights(len(self.perceptrons))

    # training algorithm
        for i in range(self.epochs):
          r_i = np.random.randint(0, num_row)   # <-- riga corretta
        row = train_set[r_i]
        y_atteso = self.predict(row)
        target = y[r_i]

            # Calculate update for the output layer
        self.output_neuron.calculate_update(self.eta, target)

            # Calculate update for the hidden layer
        for neuron in self.perceptrons:
                neuron.calculate_update(self.eta, target)
            
            # Update the output layer
        self.output_neuron.update_neuron()

            # Update the hidden layer
        for neuron in self.perceptrons:
                neuron.update_neuron()

                # At every 100 epochs calculate the error of the
                # whole training set
                if i % 100 == 0:
                    total_error = 0
                for r_i in range(num_row):
                    row = train_set[r_i]
                    y_atteso = self.predict(row)
                    error = (y[r_i]-y_atteso)
                    total_error += (error**2)*0.5
num_neuron = 3
eta = 0.01
epochs = 400
classifier =MultiLayerPerceptron(num_neuron, eta, epochs)
classifier.fit(train_set, y)


