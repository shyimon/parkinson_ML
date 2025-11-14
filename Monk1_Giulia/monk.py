from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
x = monk_s_problems.data.features 
y = monk_s_problems.data.targets 

# metadata 
print(monk_s_problems.metadata) 
  
# variable information 
print(monk_s_problems.variables) 

print("Dimensione del dataset:", len(x))

x = x.to_numpy()
y = y.to_numpy().astype(float).ravel()

# Data split
number_train = int(len(x) * 0.5)
print("How many training data:", number_train)
number_validation = int((len(x) - number_train) * 0.5)
number_test = int((len(x) - number_train) * 0.5)
print("How many validation data:", number_validation)
print("How many test data:", number_test)

train_set = x[:number_train]
validation_set = x[number_train:number_train + number_validation]
test_set = x[number_train + number_validation:]
y_train = y[:number_train]
y_val   = y[number_train:number_train + number_validation]
y_test  = y[number_train + number_validation:]

# trasformo i target in colonne (N,1) per i calcoli vettoriali
y_train_col = y_train.reshape(-1, 1)
y_val_col   = y_val.reshape(-1, 1)
y_test_col  = y_test.reshape(-1, 1)

# Neural network: input -> 2 hidden layer -> output
input_dim = train_set.shape[1] #6 features
hidden_dim = 2
output_dim = 1

rng = np.random.default_rng(0)
# Parameters definition and initialitation (hidden layer)
w_hidd = rng.uniform(0.01, 0.1, size=(input_dim, hidden_dim))  # (6, hidden_dim)
b_hidd = np.zeros((1, hidden_dim))
# Parameters definition and initialitation (output unit)
w_out = rng.uniform(0.01, 0.1, size=(hidden_dim, output_dim))  # (hidden_dim, 1)
b_out = np.zeros((1, 1))

# Hyperparameters
eta = 0.1
epochs = 1000

# activation functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def derivative_sigmoid(a):
    return a * (1 - a)

#Backpropagation learning algorithm
for epoch in range(epochs):
    # FORWARD PASS
    # hidden layer 
    net_1 = train_set @ w_hidd + b_hidd    
    out_1 = sigmoid(net_1)                  

    # output layer
    net_2 = out_1 @ w_out + b_out           
    out_2 = sigmoid(net_2)                  

    # Mean square error
    diff = out_2 - y_train_col              
    loss = np.mean(diff ** 2)

    # BACKWARD PASS
    N = len(train_set)

    # dL/d(out_2) = 2*(out_2 - y)/N
    dL_d_out2 = 2 * diff / N                # (N,1)

    # d(out_2)/d(net_2) = sigmoid'(net_2) = out_2*(1-out_2)
    dL_d_net2 = dL_d_out2 * derivative_sigmoid(out_2) 

    # Gradients for w_out and b_out
    dL_dw_out = out_1.T @ dL_d_net2        
    dL_db_out = np.sum(dL_d_net2, axis=0, keepdims=True) 

    # Backpropagation to the hidden layer
    dL_d_out1 = dL_d_net2 @ w_out.T        
    dL_d_net1 = dL_d_out1 * derivative_sigmoid(out_1)  

    dL_dw_hidd = train_set.T @ dL_d_net1  
    dL_db_hidd = np.sum(dL_d_net1, axis=0, keepdims=True) 

    # UPDATE OF THE PARAMETERS 
    w_out  -= eta * dL_dw_out
    b_out  -= eta * dL_db_out
    w_hidd -= eta * dL_dw_hidd
    b_hidd -= eta * dL_db_hidd

# Training set accuracy
net_1 = train_set @ w_hidd + b_hidd
out_1 = sigmoid(net_1)

net_2 = out_1 @ w_out + b_out
final_output = sigmoid(net_2)
final_output = (final_output >= 0.5).astype(int)

comparison = (final_output == y_train_col)
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
acc_train = np.mean(pred_train == y_train_col) * 100

pred_val = predict(validation_set)
acc_val = np.mean(pred_val == y_val_col) * 100

pred_test = predict(test_set)
acc_test = np.mean(pred_test == y_test_col) * 100

print("\nFINAL ACCURACY")
print(f"Train accuracy:      {acc_train:.2f} %")
print(f"Validation accuracy: {acc_val:.2f} %")
print(f"Test accuracy:       {acc_test:.2f} %")