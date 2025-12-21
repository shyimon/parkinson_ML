import numpy as np
import data_manipulation as data
from cascade_correlation import CascadeNetwork
import pandas as pd
import matplotlib.pyplot as plt

tr_X, tr_y, val_X, val_y, te_X, te_y = data.return_CUP()

cup_train_X = np.vstack((tr_X, val_X))
cup_train_y = np.vstack((tr_y, val_y))

cup_test_X = te_X
cup_test_y = te_y

cup_train_X, _, cup_test_X = data.normalize_dataset(cup_train_X, cup_train_X, cup_test_X, min_val=-1, max_val=1)
cup_train_y, _, cup_test_y = data.normalize_dataset(cup_train_y, cup_train_y, cup_test_y, min_val=-1, max_val=1)

n_inputs = cup_train_X.shape[1]
n_outputs = cup_train_y.shape[1]

eta = 0.7 

print(f"Creating Cascade Network: {n_inputs} Inputs -> {n_outputs} Outputs")
print("Algorithm: Quickprop")

net = CascadeNetwork(n_inputs, n_outputs, learning_rate=0.05, algorithm='quickprop')

print("Start training Phase 0 (Linear Training)...")

final_error = net.train(cup_train_X, cup_train_y, max_epochs=2000, tolerance=0.001, patience=30, max_hidden_units=10)

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


