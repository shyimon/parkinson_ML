import numpy as np
import data_manipulation as data
import neural_network as nn


X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk2(one_hot=True, dataset_shuffle=True)

# Normalization
# X_train_normalized = data.normalize(X_train, 0, 1, X_train.min(axis=0), X_train.max(axis=0))
# X_test_normalized = data.normalize(X_test, 0, 1, X_train.min(axis=0), X_train.max(axis=0))
X_train_normalized = X_train
X_val_normalized = X_val
X_test_normalized = X_test

network_structure = [X_train_normalized.shape[1]]
network_structure.append(6)  # Hidden layer with 4 neurons
network_structure.append(1)  # Output layer with 1 neuron
eta = 0.7        # Learning rate

# Network is created and trained
print("Creating neural network with huber loss...")
net = nn.NeuralNetwork(network_structure, eta=eta, loss_type="half_mse", l2_lambda=0.000001, algorithm="sgd", activation_type="sigmoid", eta_plus=1.2, eta_minus=0.5, mu=1.75, decay=0.9, weight_initialzer="def", momentum=0.9)
print("Start training with rprop...")
best_val = net.fit(X_train_normalized, y_train, X_val_normalized, y_val, epochs=1000, batch_size=16, patience=32)

# Accuracy for the training set
print(f"\nBest validation: {best_val:.6f}\nCalculating accuracy...")
y_pred = net.predict(X_train_normalized)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)

accuracy = np.mean(y_pred_class == y_train) * 100
print(f"\nFinal Training Accuracy: {accuracy:.2f}%")

# Accuracy for the test set
y_pred_test = net.predict(X_test_normalized)
y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
test_accuracy = np.mean(y_pred_test_class == y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

print(f"Details:")
print(f"Correctly predicted training patterns: {np.sum(y_pred_class == y_train)}/{len(y_train)}")
print(f"Correctly predicted test patterns: {np.sum(y_pred_test_class == y_test)}/{len(y_test)}")
net.save_plots("img/plot.png")
net.draw_network("img/network")