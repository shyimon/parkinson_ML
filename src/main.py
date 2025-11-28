from ucimlrepo import fetch_ucirepo 
import numpy as np

# Custom functions and classes
import data_manipulation as data
import neural_network as nn

# === MAIN EXECUTION ===
# Carica i dati
X_train, y_train, X_test, y_test = data.return_monk1()

# Normalizza i dati
X_train_normalized = data.normalize(X_train, -1, 1)
X_test_normalized = data.normalize(X_test, -1, 1)

# Parametri della rete
num_inputs = X_train_normalized.shape[1]
num_hidden = 4      # Ridotto per MONK-1
num_outputs = 1
eta = 0.05           # Learning rate aumentato

# Crea e allena la rete 
mlp = nn.NeuralNetwork([6, 4, 1], eta=eta)
print("Inizio training...")
mlp.fit(X_train_normalized, X_test_normalized, y_train, y_test, epochs=500)

# Predizioni e accuracy
print("\nCalcolo accuracy...")
y_pred = mlp.predict(X_train_normalized)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)

accuracy = np.mean(y_pred_class == y_train) * 100
print(f"\nFinal Training Accuracy: {accuracy:.2f}%")

# Test accuracy
y_pred_test = mlp.predict(X_test_normalized)
y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
test_accuracy = np.mean(y_pred_test_class == y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Dettagli aggiuntivi
print(f"\nDettagli:")
print(f"Pattern corretti: {np.sum(y_pred_class == y_train)}/{len(y_train)}")
print(f"Pattern test corretti: {np.sum(y_pred_test_class == y_test)}/{len(y_test)}")
mlp.save_plots("img/plot.png")