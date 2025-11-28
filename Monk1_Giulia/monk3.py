from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import math
import itertools

def return_monk1():
    train_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train'
    test_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    train_set = pd.read_csv(train_set_url, header=None, names=column_names, delim_whitespace=True)
    test_set = pd.read_csv(test_set_url, header=None, names=column_names, delim_whitespace=True)

    # Separazione corretta
    X_train = train_set.drop(columns=['class', 'id']).to_numpy()
    y_train = train_set['class'].to_numpy()
    
    X_test = test_set.drop(columns=['class', 'id']).to_numpy()
    y_test = test_set['class'].to_numpy()
    
    return X_train, y_train, X_test, y_test

class Neuron:
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron
        self.weights = np.random.uniform(-0.5, 0.5, size=num_inputs)
        self.bias = np.random.uniform(-0.5, 0.5) # la scelta del valore del bias è da giustificare con una grid search
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.inputs = None
        self.in_output_neurons = []

    def attach_to_output(self, neurons):
        self.in_output_neurons = list(neurons)

    def sigmoid(self, x):
        # Aggiunta di clipping per evitare overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + math.exp(-x))

    def derivative_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def predict(self, inputs):
        inputs = np.array(inputs, dtype=float)
        self.inputs = inputs
        self.net = float(np.dot(self.weights, inputs)) + self.bias
        self.output = self.sigmoid(self.net)
        return self.output

    def compute_delta_output(self, target):
        self.delta = (target - self.output) * self.derivative_sigmoid(self.net)
    
    def compute_delta_hidden(self):
        delta_sum = 0.0
        for k in self.in_output_neurons:
            w_kj = k.weights[self.index_in_layer]
            delta_sum += k.delta * w_kj
        self.delta = delta_sum * self.derivative_sigmoid(self.net)

    def update_weights(self, eta):
        self.weights += eta * self.delta * self.inputs
        self.bias += eta * self.delta

class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, num_outputs=1, eta=0.2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.eta = eta
        self.loss_history = []

        self.hidden_layer = [Neuron(num_inputs=num_inputs, index_in_layer=j) for j in range(num_hidden)]
        self.output_layer = [Neuron(num_inputs=num_hidden, index_in_layer=k, is_output_neuron=True) for k in range(num_outputs)]

        for h in self.hidden_layer:
            h.attach_to_output(self.output_layer)

    def forward(self, x):
        hidden_outputs = [h.predict(x) for h in self.hidden_layer]
        outputs = [o.predict(hidden_outputs) for o in self.output_layer]
        return hidden_outputs, outputs
    
    def backward(self, x, y_true):
        if self.num_outputs == 1:
            y_true = [y_true]

        for k, o in enumerate(self.output_layer):
            o.compute_delta_output(y_true[k])

        for h in self.hidden_layer:
            h.compute_delta_hidden()

        hidden_outputs = [h.output for h in self.hidden_layer]
        for o in self.output_layer:
            o.inputs = np.array(hidden_outputs, dtype=float)
            o.update_weights(self.eta)

        for h in self.hidden_layer:
            h.update_weights(self.eta)
   
    def fit(self, X, y, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for epoch in range(epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                _, outputs = self.forward(xi)
                y_pred = outputs[0]
                total_loss += 0.5 * (yi - y_pred) ** 2
                self.backward(xi, yi)

            if epoch % 100 == 0:
                avg_loss = total_loss / len(X)
                self.loss_history.append(avg_loss)
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        preds = []
        for xi in X:
            _, outputs = self.forward(xi)
            preds.append(outputs[0])
        return np.array(preds)

def calculate_accuracy(mlp, X, y):
    y_pred = mlp.predict(X)
    y_pred_class = np.where(y_pred >= 0.5, 1, 0)
    accuracy = np.mean(y_pred_class == y) * 100
    return accuracy
    
def create_k_folds(X, y, k_folds, random_seed=42):
    #crea k fold per la cross validation
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k_folds

    folds = []
    for i in range(k_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k_folds - 1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
    
    return folds

def print_grid_search_header(param_grid):
    """Stampa l'header della grid search"""
    print("\n" + "=" * 80)
    print("GRID SEARCH - HYPERPARAMETER TUNING")
    print("=" * 80)
    
    for param_name, param_values in param_grid.items():
        print(f"Hyperparameter '{param_name}': {param_values}")
    
    print("=" * 80)

def print_accuracy_grid(results_matrix, hidden_neurons, learning_rates, fixed_params):
    """Stampa la griglia delle accuracy in formato tabellare"""
    print(f"\nGRID SEARCH RESULTS (Accuracy %)")
    print(f"Fixed parameters: {fixed_params}")
    print("-" * 60)
    
    # Header colonne (learning rates)
    header = "Hidden \\ Eta | " + " | ".join([f"{eta:5.2f}" for eta in learning_rates])
    print(header)
    print("-" * len(header))
    
    # Righe (hidden neurons)
    for i, hidden in enumerate(hidden_neurons):
        row = f"    {hidden:2d}     | " + " | ".join([f"{results_matrix[i][j]:5.1f}%" for j in range(len(learning_rates))])
        print(row)
    
    print("-" * len(header))

def grid_search_mlp(X_train, y_train, param_grid, k_folds=3, random_seed=42):
    num_inputs = X_train.shape[1]
    best_params = None
    best_accuracy = 0
    all_results = []
    
    print_grid_search_header(param_grid)

    # Per stampare la griglia, assumiamo che stiamo facendo una ricerca 2D
    # (ad esempio hidden_neurons vs learning_rate con epochs fisso)
    if 'num_hidden' in param_grid and 'eta' in param_grid:
        hidden_neurons = param_grid['num_hidden']
        learning_rates = param_grid['eta']
        epochs_value = param_grid['epochs'][0] if 'epochs' in param_grid else 700
        
        # Matrice per i risultati
        results_matrix = np.zeros((len(hidden_neurons), len(learning_rates)))
        
        print(f"\nTesting {len(hidden_neurons)}x{len(learning_rates)} = {len(hidden_neurons)*len(learning_rates)} combinations...")
    
    # Genera tutte le combinazioni di parametri
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    print(f"Grid Search: testing {len(param_combinations)} combinations with {k_folds}-fold CV")
    print("=" * 70)
    
    for i, combo in enumerate(param_combinations):
        params = dict(zip(param_names, combo))
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        fold_accuracies = []
        
        # Crea i fold per cross-validation
        folds = create_k_folds(X_train, y_train, k_folds, random_seed)
        
        for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
            np.random.seed(random_seed + fold_idx) 
            # Crea e allena il modello con i parametri correnti
            mlp = MultiLayerPerceptron(num_inputs=num_inputs, num_hidden=params['num_hidden'], num_outputs=1, eta=params['eta'])
            
            mlp.fit(X_train_fold, y_train_fold, epochs=params['epochs'])
            
            # Calcola accuracy sul validation set
            val_accuracy =calculate_accuracy(mlp, X_val_fold, y_val_fold)
            fold_accuracies.append(val_accuracy)
        
        # Calcola media e deviazione standard delle accuracy
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        
        # Salva i risultati
        result = {
            'params': params.copy(),
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_accuracies': fold_accuracies
        }
        all_results.append(result)
        
        # Aggiorna la matrice per la visualizzazione a griglia
        if 'num_hidden' in param_grid and 'eta' in param_grid:
            hidden_idx = hidden_neurons.index(params['num_hidden'])
            eta_idx = learning_rates.index(params['eta'])
            results_matrix[hidden_idx][eta_idx] = mean_accuracy
        
        # Aggiorna i migliori parametri
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params.copy()
    
    # Stampa la griglia dei risultati
    if 'num_hidden' in param_grid and 'eta' in param_grid:
        fixed_params = {'epochs': epochs_value}
        print_accuracy_grid(results_matrix, hidden_neurons, learning_rates, fixed_params)
    
    print("=" * 80)
    print(f"Best parameters: {best_params}")
    print(f"Best CV accuracy: {best_accuracy:.2f}%")
    
    return best_params, best_accuracy, all_results

def print_top_results(all_results, top_k=5):
    print(f"\nTop {top_k} parameter combinations:")
    print("-" * 60)
    
    # Ordina i risultati per accuracy discendente
    sorted_results = sorted(all_results, key=lambda x: x['mean_accuracy'], reverse=True)
    
    for i, result in enumerate(sorted_results[:top_k]):
        params = result['params']
        mean_acc = result['mean_accuracy']
        std_acc = result['std_accuracy']
        print(f"{i+1}. Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
        print(f"   Parameters: {params}")
        print()

# === MAIN EXECUTION ===
def main():
    # Carica i dati
    X_train, y_train, X_test, y_test = return_monk1()

    # Definisci la grid degli hyperparameters
    param_grid = {'num_hidden': [2, 4, 6, 8],'eta': [0.1, 0.2, 0.3, 0.5],'epochs': [500, 700, 1000]}
    
    # Esegui grid search
    print("Starting Grid Search for Hyperparameter Tuning...")
    best_params, best_accuracy, all_results = grid_search_mlp(X_train, y_train, param_grid, k_folds=3, random_seed=42)
    
    # Mostra i migliori risultati
    print_top_results(all_results, top_k=5)
    
     #alleno il modello finale con i migliori parametri
    print("Training final model with best parameters...")
    np.random.seed(42)
    final_mlp = MultiLayerPerceptron(num_inputs=X_train.shape[1], num_hidden=best_params['num_hidden'], num_outputs=1, eta=best_params['eta'])

    final_mlp.fit(X_train, y_train, epochs = best_params['epochs'])

    # Valutazione finale
    y_pred_train = final_mlp.predict(X_train)
    y_pred_train_class = np.where(y_pred_train >= 0.5, 1, 0)
    train_accuracy = np.mean(y_pred_train_class == y_train) * 100
    
    y_pred_test = final_mlp.predict(X_test)
    y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
    test_accuracy = np.mean(y_pred_test_class == y_test) * 100
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best parameters: {best_params}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
        main()