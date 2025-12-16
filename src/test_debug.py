import numpy as np
import data_manipulation as data
import neural_network as nn

# Carica Monk3
X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk3(one_hot=True, dataset_shuffle=True)

# Normalizza
X_train_norm, X_val_norm, X_test_norm = data.normalize_dataset(X_train, X_val, X_test, 0, 1)

# Crea una rete molto semplice
net = nn.NeuralNetwork(
    network_structure=[X_train.shape[1], 4, 1],  # input, hidden, output
    eta=0.1,  # Learning rate più alto per test
    loss_type="half_mse",
    algorithm="sgd",
    activation_type="sigmoid",
    l2_lambda=0.0,
    momentum=0.0,
    debug=True  # Attiva debug
)

# Test forward pass
print("Test forward pass:")
sample = X_train_norm[0]
print(f"Input shape: {sample.shape}")
output = net.forward(sample)
print(f"Output: {output}")
print(f"Target: {y_train[0]}")

# Test backward pass
print("\nTest backward pass:")
err = output - y_train[0]
print(f"Error: {err}")
net.backward(err, accumulate=False)

# Test training su 10 epoche
print("\nTest training su 10 epoche:")
history = net.fit(
    X_train_norm[:50], y_train[:50],  # Usa solo 50 esempi per test
    X_val_norm[:10], y_val[:10],
    epochs=10,
    batch_size=4,
    patience=5,
    verbose=True
)

# Test accuracy
y_pred = net.predict(X_val_norm[:10])
y_pred_class = np.where(y_pred >= 0.5, 1, 0)
accuracy = np.mean(y_pred_class == y_val[:10]) * 100
print(f"\nAccuracy su validation (10 esempi): {accuracy:.2f}%")