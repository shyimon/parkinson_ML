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
train_set = train_set.astype(float).to_numpy()
test_set  = test_set.astype(float).to_numpy()
y = y.astype(int).to_numpy().ravel()
print("I tipi di variabili train set e test set adesso sono:", type(train_set), type (test_set))
print("Il tipo di variabile y adesso è:", type(y))
print(train_set)
train_set = np.delete(train_set, 7, axis=1)
print(train_set)

y_train = y[:len(train_set)]
y_test = y[:len(test_set)]

# Parameters definition and initialitation (hidden layer)
train_coloumns = np.shape(train_set)[1]
num_hidd_neur = 1
num_out_neur = 1
w_hidd = np.random.uniform(0.01, 0.001, size=(train_coloumns, num_hidd_neur))
b_hidd = np.random.uniform(0.01, 0.001, size=(num_hidd_neur))
w_out = np.random.uniform(0.01, 0.001, size=(num_out_neur, 1))  
b_out = np.random.uniform(0.01, 0.001)

# activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def derivative_sigmoid(a):
    return - 1.0 / ((1.0 + np.exp(-a)) ** 2)

# Hyperparameters
eta = 0.5
epochs = 390

#Backpropagation learning algorithm
# Neural network: input -> 1 hidden neuron -> 1 output neuron
for epoch in range(epochs):
    # FORWARD PASS
    # hidden layer 
    net_hidd = train_set @ w_hidd + b_hidd  
    out_hidd = sigmoid(net_hidd)                  

    # output layer
    net_out = out_hidd @ w_out + b_out           
    final_out = sigmoid(net_out)                  

    # Mean square error
    loss_pattern =  0.5 * ((y_train - final_out)**2)

    # BACKWARD PASS
    # delta output neuron
    dE_dnet_out = final_out - y_train 
    derivative_out = derivative_sigmoid(final_out)   
    delta_out = dE_dnet_out * derivative_out         
    # Gradients for w_out and b_out
    grad_w_out = np.sum(delta_out * out_hidd)    
    grad_b_out = np.sum(delta_out)
    w_out -= eta * grad_w_out
    b_out -= eta * grad_b_out
    
    # delta output neuron
    dE_dout_hidd = delta_out * w_out
    derivative_hidd = derivative_sigmoid(out_hidd)   
    delta_hidd = dE_dout_hidd * derivative_hidd         
    # Gradients for w_hidd and b_hidd
    grad_w_hidd = (np.transpose(train_set)) @ delta_hidd   
    grad_b_hidd= np.sum(delta_hidd)
    w_hidd -= eta * grad_w_hidd
    b_hidd -= eta * grad_b_hidd
    

# Training set accuracy
net_1 = train_set @ w_hidd + b_hidd
out_1 = sigmoid(net_1)

net_2 = out_1 @ w_out + b_out
final_output = sigmoid(net_2)
final_output = (final_output >= 5).astype(int)

comparison = (final_output == y_train)
num_correct = np.sum(comparison)
num_wrong = len(train_set) - num_correct
accuracy = num_correct / len(train_set) * 100

print("\nTRAIN SET")
print("Correct:", num_correct)
print("Wrong:", num_wrong)
print("Accuracy:", accuracy, "%")

# Prediction function
def predict(x_set):
    net_1 = x_set @ w_hidd + b_hidd
    out_1 = sigmoid(net_1)
    net_2 = out_1 @ w_out + b_out
    out_2 = sigmoid(net_2)
    return (out_2 >= 0.5).astype(int)

# Accuracy of train/validation/test
pred_train = predict(train_set)
acc_train = np.mean(pred_train == y_train) * 100


pred_test = predict(test_set)
acc_test = np.mean(pred_test == y_test) * 100

print("\nFINAL ACCURACY")
print(f"Train accuracy:      {acc_train:.2f} %")
print(f"Test accuracy:       {acc_test:.2f} %")
