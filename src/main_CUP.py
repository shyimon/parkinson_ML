import numpy as np
import data_manipulation as data
import neural_network as nn
import pandas as pd
import matplotlib.pyplot as plt


cup_train_X, cup_train_y, cup_val_X, cup_val_y, cup_test_X, cup_test_y = data.return_CUP()

train_min = cup_train_X.min(axis=0) 
train_max = cup_train_X.max(axis=0)

activation_type = "tanh"

if activation_type=="tanh":
    target_min = -1
    target_max = 1
elif activation_type=="sigmoid":
    target_min = 0
    target_max = 1

cup_test_X = data.normalize(cup_test_X, target_min, target_max, train_min, train_max)
cup_test_y = data.normalize(cup_test_y, target_min, target_max, cup_train_y.min(axis=0), cup_train_y.max(axis=0))
cup_val_X = data.normalize(cup_val_X, target_min, target_max, train_min, train_max)
cup_val_y = data.normalize(cup_val_y, target_min, target_max, cup_train_y.min(axis=0), cup_train_y.max(axis=0))
cup_train_X = data.normalize(cup_train_X, target_min, target_max, train_min, train_max)
cup_train_y = data.normalize(cup_train_y, target_min, target_max, cup_train_y.min(axis=0), cup_train_y.max(axis=0))

# np.savetxt("cuptest.csv", cup_test_X, delimiter=",", fmt='%.4f')

network_structure = [cup_test_X.shape[1]]
network_structure.append(6)
network_structure.append(4)
network_structure.append(cup_test_y.shape[1])
eta = 0.1

print("Creating neural network with huber loss...")
net = nn.NeuralNetwork(network_structure, eta=eta, loss_type="half_mse", l2_lambda=0.0000001, algorithm='sgd', activation_type=activation_type, eta_plus=1.2, eta_minus=0.5, weight_initializer="def")
print("Start training...")
net.fit(cup_train_X, cup_train_y, cup_val_X, cup_val_y, epochs=1000, batch_size=20, patience=20)

y_pred = net.predict(cup_train_X)
training_errors =  0.5 * (y_pred - cup_train_y) ** 2 # half mse
print(f"The mean TRAINING squared errors for each feature is: {np.mean(training_errors, axis=0)}")
print(f"The total TRAINING mean squared error is: {np.mean(np.mean(training_errors, axis=0))}")

y_pred = net.predict(cup_test_X)
training_errors =  0.5 * (y_pred - cup_test_y) ** 2 # half mse
print(f"\nThe mean TEST squared errors for each feature is: {np.mean(training_errors, axis=0)}")
print(f"The total TEST mean squared error is: {np.mean(np.mean(training_errors, axis=0))}")

net.save_plots("img/cup_plot.png")
net.draw_network("img/cup_network")