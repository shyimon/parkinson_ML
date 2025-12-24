import numpy as np
import data_manipulation as data
from cascade_correlation import CascadeNetwork
import matplotlib.pyplot as plt

tr_X, tr_y, val_X, val_y, te_X, te_y = data.return_CUP() # Caricamento dati CUP

cup_train_X = np.vstack((tr_X, val_X)) 
cup_train_y = np.vstack((tr_y, val_y)) 

cup_test_X = te_X
cup_test_y = te_y

x_min = cup_train_X.min(axis=0)
x_max = cup_train_X.max(axis=0)

cup_train_X = data.normalize(cup_train_X, -1, 1, x_min, x_max)
cup_test_X = data.normalize(cup_test_X, -1, 1, x_min, x_max)

y_min = cup_train_y.min(axis=0)
y_max = cup_train_y.max(axis=0)

cup_train_y = data.normalize(cup_train_y, -1, 1, y_min, y_max)
cup_test_y = data.normalize(cup_test_y, -1, 1, y_min, y_max)

n_inputs = cup_train_X.shape[1]
n_outputs = cup_train_y.shape[1]

eta = 0.05
patience = 30
max_units = 10

print(f"Creating Cascade Network: {n_inputs} Inputs -> {n_outputs} Outputs")
print("Algorithm: Quickprop")

net = CascadeNetwork(n_inputs, n_outputs, eta, algorithm='quickprop')

print("Start training Phase 0 (Linear Training)...")

final_error = net.train(cup_train_X, cup_train_y, X_val=cup_test_X, y_val=cup_test_y, max_epochs=2000, tolerance=0.001, patience=30, max_hidden_units=10)

print(f"Phase 0 ended. Residual Error: {final_error:.5f}")

print("\nCalculating Test Set Error...")
test_error = 0.0
for i in range(len(cup_test_X)):
    
    pattern = cup_test_X[i]
    target = cup_test_y[i]
    
    output = net.forward(pattern)

    test_error += np.sum(0.5 * (target - output) ** 2)

avg_test_error = test_error / len(cup_test_X)
print(f"Mean Test Error (MSE): {avg_test_error:.5f}")
avg_train_error = final_error / len(cup_train_X)
print(f"Mean Training Error (MSE): {avg_train_error:.5f}")

# net.save_plots("img/cup_plot.png")
# net.draw_network("img/cup_network")


