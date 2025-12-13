import numpy as np
import data_manipulation as data
from cascade_correlation import CascadeNetwork
import pandas as pd
import matplotlib.pyplot as plt

cup_train_X, cup_train_y, cup_test_X, cup_test_y = data.return_CUP()

cup_test_X = data.normalize(cup_test_X, -1, 1, cup_train_X.min(axis=0), cup_train_X.max(axis=0))
cup_test_y = data.normalize(cup_test_y, -1, 1, cup_train_y.min(axis=0), cup_train_y.max(axis=0))
cup_train_X = data.normalize(cup_train_X, -1, 1, cup_train_X.min(axis=0), cup_train_X.max(axis=0))
cup_train_y = data.normalize(cup_train_y, -1, 1, cup_train_y.min(axis=0), cup_train_y.max(axis=0))

n_inputs = cup_train_X.shape[1]
n_outputs = cup_train_y.shape[1]

eta = 0.7 

print(f"Creating Cascade Network: {n_inputs} Inputs -> {n_outputs} Outputs")
print("Algorithm: Quickprop")

net = CascadeNetwork(n_inputs, n_outputs, learning_rate=eta, algorithm='quickprop')

print("Start training Phase 0 (Linear Training)...")

final_error = net.train_phase_0(cup_train_X, cup_train_y, epochs=1000, tolerance=0.001, patience=50)

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

# net.save_plots("img/cup_plot.png")
# net.draw_network("img/cup_network")

