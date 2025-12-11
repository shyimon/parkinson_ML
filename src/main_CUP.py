import numpy as np
import data_manipulation as data
import neural_network as nn
import pandas as pd
import matplotlib.pyplot as plt


cup_train_X, cup_train_y, cup_test_X, cup_test_y = data.return_CUP()

cup_test_X = data.normalize(cup_test_X, -1, 1, cup_train_X.min(axis=0), cup_train_X.max(axis=0))
cup_test_y = data.normalize(cup_test_y, -1, 1, cup_train_y.min(axis=0), cup_train_y.max(axis=0))
cup_train_X = data.normalize(cup_train_X, -1, 1, cup_train_X.min(axis=0), cup_train_X.max(axis=0))
cup_train_y = data.normalize(cup_train_y, -1, 1, cup_train_y.min(axis=0), cup_train_y.max(axis=0))

network_structure = [cup_test_X.shape[1]]
network_structure.append(8)
network_structure.append(4)
network_structure.append(cup_test_y.shape[1])
eta = 0.05

print("Creating neural network with huber loss...")
net = nn.NeuralNetwork(network_structure, eta=eta, loss_type="half_mse", l2_lambda=0.0001, algorithm='sgd', activation_type="tanh", eta_plus=1.2, eta_minus=0.5, weight_initializer="def")
print("Start training...")
net.fit(cup_train_X, cup_test_X, cup_train_y, cup_test_y, epochs=400, batch_size=25, patience=20)

y_pred = net.predict(cup_train_X)


net.save_plots("img/cup_plot.png")
net.draw_network("img/cup_network")