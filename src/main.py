import numpy as np
import data_manipulation as data
import neural_network as nn


X_train, y_train, X_test, y_test = data.return_monk3(one_hot=True)

# Normalization
# X_train_normalized = data.normalize(X_train, 0, 1)
# X_test_normalized = data.normalize(X_test, 0, 1)
X_train_normalized = X_train
X_test_normalized = X_test


network_structure = [X_train_normalized.shape[1]]
network_structure.append(4)  # Hidden layer with 4 neurons
network_structure.append(1)  # Output layer with 1 neuron
eta = 0.7           # Learning rate


nn = nn.NeuralNetwork(network_structure, eta=eta)
print("Start training...")
nn.fit(X_train_normalized, X_test_normalized, y_train, y_test, epochs=500)

# Accuracy for the training set
print("\nCalculating accuracy...")
y_pred = nn.predict(X_train_normalized)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)

accuracy = np.mean(y_pred_class == y_train) * 100
print(f"\nFinal Training Accuracy: {accuracy:.2f}%")

# Accuracy for the test set
y_pred_test = nn.predict(X_test_normalized)
y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
test_accuracy = np.mean(y_pred_test_class == y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")


print(f"Details:")
print(f"Correctly predicted training patterns: {np.sum(y_pred_class == y_train)}/{len(y_train)}")
print(f"Correctly predicted test patterns: {np.sum(y_pred_test_class == y_test)}/{len(y_test)}")
nn.save_plots("img/plot.png")