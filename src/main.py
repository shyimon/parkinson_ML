from ucimlrepo import fetch_ucirepo 
import numpy as np

# Custom functions and classes
import data_manipulation as data
import neural_network as nn

# === MAIN EXECUTION ===
# Loading of data
X_train, y_train, X_test, y_test = data.return_monk1()

# Normalization
X_train_normalized = data.normalize(X_train, -1, 1)
X_test_normalized = data.normalize(X_test, -1, 1)

# Definition of network structure
network_structure = [X_train_normalized.shape[1]]
network_structure.append(4)  # Hidden layer with 4 neurons
network_structure.append(1)  # Output layer with 1 neuron
eta = 0.05           # Learning rate

# Network is created and trained
nn = nn.NeuralNetwork(network_structure, eta=eta)
print("Start training...")
nn.fit(X_train_normalized, X_test_normalized, y_train, y_test, epochs=500)

# Accuracy is computed for the training set
print("\\Calculating accuracy...")
y_pred = nn.predict(X_train_normalized)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)

accuracy = np.mean(y_pred_class == y_train) * 100
print(f"\nFinal Training Accuracy: {accuracy:.2f}%")

# Accuracy is computed for the test set
y_pred_test = nn.predict(X_test_normalized)
y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
test_accuracy = np.mean(y_pred_test_class == y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Detailed results
print(f"Details:")
print(f"Correctly predicted training patterns: {np.sum(y_pred_class == y_train)}/{len(y_train)}")
print(f"Correctly predicted test patterns: {np.sum(y_pred_test_class == y_test)}/{len(y_test)}")
nn.save_plots("img/plot.png")