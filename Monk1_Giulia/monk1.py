from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import math
import itertools
from sklearn.metrics import accuracy_score, classification_report
import time

def return_monk1():
    train_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
    test_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    train_set = pd.read_csv(train_set_url, header=None, names=column_names, delim_whitespace=True)
    test_set = pd.read_csv(test_set_url, header=None, names=column_names, delim_whitespace=True)

    # Separazione corretta
    X_train = train_set.drop(columns=['class', 'id']).to_numpy()
    y_train = train_set['class'].to_numpy()
    
    X_test = test_set.drop(columns=['class', 'id']).to_numpy()
    y_test = test_set['class'].to_numpy()
    
    return X_train, y_train, X_test, y_test

class CascadeCorrelation:
    def __init__(self, input_size, activation_function='tanh', weight_init_range=(-0.1, 0.1), use_bias=True):
        self.input_size = input_size
        self.activation_function_name = activation_function
        self.activation_function = self._get_activation_function(activation_function)
        self.weight_init_range = weight_init_range
        self.use_bias = use_bias
        
        # Inizializza i pesi nel costruttore
        self._initialize_weights(input_size)
        
        # Altri attributi
        self.hidden_units = []
        self.hidden_weights = []
        self.max_hidden_units = 5  # Valore di default, verrà sovrascritto
        self.learning_rate = 0.1
        self.training_history = []
    
    def _initialize_weights(self, input_size):
        """Inizializza i pesi dello strato di output"""
        low, high = self.weight_init_range
        self.output_weights = np.random.uniform(low, high, (1, input_size))
        
        if self.use_bias:
            self.output_bias = np.random.uniform(low, high, 1)
        else:
            self.output_bias = np.array([0.0])
    
    def _get_activation_function(self, name):
        """Restituisce la funzione di attivazione"""
        if name == 'tanh':
            return np.tanh
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        else:
            return np.tanh
    
    def _compute_output(self, x):
        """Calcola l'output della rete"""
        x = np.array(x)
        
        # Converti in array 2D se necessario
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Controllo e correzione dimensioni
        if x.shape[1] != self.output_weights.shape[1]:
            if x.shape[1] == self.output_weights.shape[1] - 1 and self.use_bias:
                x = np.column_stack([x, np.ones(x.shape[0])])
            elif x.shape[1] == self.output_weights.shape[1] + 1 and not self.use_bias:
                x = x[:, :-1]
            elif x.shape[1] > self.output_weights.shape[1]:
                x = x[:, :self.output_weights.shape[1]]
            else:
                temp = np.zeros((x.shape[0], self.output_weights.shape[1]))
                temp[:, :x.shape[1]] = x
                x = temp
        
        # Calcola l'output
        net = np.dot(self.output_weights, x.T) + self.output_bias.reshape(-1, 1)
        output = self.activation_function(net)
        
        return output.flatten()
    
    def _add_hidden_unit(self, X, y):
        """Aggiunge una nuova unità nascosta"""
        low, high = self.weight_init_range
        candidate_weights = np.random.uniform(low, high, self.input_size)
        
        if self.use_bias:
            candidate_bias = np.random.uniform(low, high, 1)
        else:
            candidate_bias = np.array([0.0])
        
        # Aggiungi il candidato alla lista delle unità nascoste
        self.hidden_units.append({
            'weights': candidate_weights,
            'bias': candidate_bias
        })
        
        # Inizializza i pesi per la nuova connessione
        hidden_weight = np.random.uniform(low, high, 1)
        self.hidden_weights.append(hidden_weight)
    
    def fit(self, X, y, max_epochs=100, tolerance=1e-4, learning_rate=0.1):
        """Addestra la rete Cascade Correlation"""
        self.learning_rate = learning_rate
        X = np.array(X)
        y = np.array(y)
        
        # Aggiungi bias term ai dati di input se use_bias è True
        if self.use_bias:
            X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        else:
            X_with_bias = X
        
        # Training dello strato di output iniziale
        previous_error = float('inf')
        training_loss = []
        
        for epoch in range(max_epochs):
            total_error = 0
            
            for i, x in enumerate(X_with_bias):
                # Calcola l'output corrente
                prediction = self._compute_output([x])[0]
                
                # Calcola l'errore
                error = y[i] - prediction
                total_error += error ** 2
                
                # Aggiorna i pesi (regola delta semplice)
                x_array = np.array(x).flatten()
                delta = error * self.learning_rate
                
                # CORREZIONE: Assicurati che le dimensioni siano compatibili
                if x_array.shape[0] != self.output_weights.shape[1]:
                    # Se c'è mismatch, aggiusta x_array
                    if x_array.shape[0] < self.output_weights.shape[1]:
                        # Aggiungi zeri
                        temp = np.zeros(self.output_weights.shape[1])
                        temp[:x_array.shape[0]] = x_array
                        x_array = temp
                    else:
                        # Tronca
                        x_array = x_array[:self.output_weights.shape[1]]
                
                # CORREZIONE: Assicurati che delta * x_array abbia la stessa dimensione di output_weights
                update = delta * x_array
                if update.shape != self.output_weights.shape:
                    update = update.reshape(self.output_weights.shape)
                
                self.output_weights += update
                
                if self.use_bias:
                    self.output_bias += delta
            
            mse = total_error / len(y)
            training_loss.append(mse)
            
            # Controllo di convergenza
            if abs(previous_error - mse) < tolerance:
                break
            
            previous_error = mse
            
            # Aggiungi unità nascoste se necessario
            if (mse > tolerance and 
                len(self.hidden_units) < self.max_hidden_units and 
                epoch % 20 == 0 and 
                epoch > 10):  # Aspetta almeno 10 epoche prima di aggiungere unità
                self._add_hidden_unit(X_with_bias, y)
        
        self.training_history = training_loss
        return training_loss[-1] if training_loss else float('inf')
    
    def predict(self, X):
        """Effettua predizioni sui dati"""
        X = np.array(X)
        
        # Aggiungi bias term se use_bias è True
        if self.use_bias:
            X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        else:
            X_with_bias = X
        
        predictions = []
        
        for i, x in enumerate(X_with_bias):
            current_input = np.array(x).flatten()
            
            # Calcola l'output dello strato iniziale
            output = self._compute_output([current_input])[0]
            
            # Aggiungi contributi delle unità nascoste
            for j, unit in enumerate(self.hidden_units):
                net = np.dot(unit['weights'], current_input) + unit['bias']
                hidden_output = self.activation_function(net)
                output += self.hidden_weights[j] * hidden_output
            
            predictions.append(output)
        
        return np.array(predictions)
    
    def predict_class(self, X, threshold=0.0):
        """Predizioni per classificazione binaria"""
        predictions = self.predict(X)
        return (predictions > threshold).astype(int)

def one_hot_encode_monk(X):
    """Codifica one-hot per i dati MONK che sono categorici"""
    encoded_features = []
    
    for i in range(6):
        feature = X[:, i]
        n_values = len(np.unique(feature))
        one_hot = np.eye(n_values)[feature - 1]
        encoded_features.append(one_hot)
    
    return np.hstack(encoded_features)

def calculate_accuracy_error(y_true, y_pred):
    """Calcola accuracy e errore associato"""
    accuracy = accuracy_score(y_true, y_pred)
    n = len(y_true)
    # Errore standard per proporzione binomiale
    error = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n)  # 95% confidence interval
    return accuracy, error

def grid_search_monk1():
    """Esegue grid search su diverse configurazioni"""
    # Carica i dati
    print("Loading MONK-1 data...")
    X_train, y_train, X_test, y_test = return_monk1()
    
    # Converti le label
    y_train_tanh = 2 * y_train - 1
    y_test_tanh = 2 * y_test - 1
    
    # Codifica one-hot
    X_train_encoded = one_hot_encode_monk(X_train)
    X_test_encoded = one_hot_encode_monk(X_test)
    
    print(f"Encoded X_train shape: {X_train_encoded.shape}")
    print(f"Encoded X_test shape: {X_test_encoded.shape}")
    
    # Calcola input_size correttamente
    input_size = X_train_encoded.shape[1]
    if True:  # use_bias sarà gestito separatamente
        input_size += 1  # +1 per il bias
    
    print(f"Network input size: {input_size}")
    
    # Definizione della grid search con hidden units da 3 a 6
    param_grid = {
        'max_hidden_units': [3, 4, 5, 6],
        'use_bias': [True, False],
        'weight_init_range': [(-0.1, 0.1), (-0.5, 0.5), (-1.0, 1.0)],
        'learning_rate': [0.01, 0.05, 0.1],
        'seed': [42, 123, 456]
    }
    
    # Genera tutte le combinazioni
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    print(f"\nStarting Grid Search with {len(combinations)} configurations...")
    print("=" * 120)
    
    results = []
    
    for i, params in enumerate(combinations, 1):
        # Imposta il seed
        np.random.seed(params['seed'])
        
        # Crea la rete
        network = CascadeCorrelation(
            input_size=input_size,
            activation_function='tanh',
            weight_init_range=params['weight_init_range'],
            use_bias=params['use_bias']
        )
        
        network.max_hidden_units = params['max_hidden_units']
        
        try:
            # Addestra la rete
            start_time = time.time()
            final_loss = network.fit(
                X_train_encoded, 
                y_train_tanh, 
                max_epochs=100, 
                tolerance=1e-4,
                learning_rate=params['learning_rate']
            )
            training_time = time.time() - start_time
            
            # Predizioni
            y_pred = network.predict_class(X_test_encoded, threshold=0.0)
            y_pred_binary = (y_pred > 0).astype(int)
            y_test_binary = (y_test > 0).astype(int)
            
            # Calcola metriche
            accuracy, accuracy_error = calculate_accuracy_error(y_test_binary, y_pred_binary)
            
            # Stampa risultati
            print(f"Config {i:3d}/{len(combinations)} | "
                  f"Loss: {final_loss:.6f} | "
                  f"Accuracy: {accuracy:.4f} ± {accuracy_error:.4f} | "
                  f"Hidden: {params['max_hidden_units']} | "
                  f"Bias: {params['use_bias']!s:5} | "
                  f"Weight Init: [{params['weight_init_range'][0]:.1f}, {params['weight_init_range'][1]:.1f}] | "
                  f"LR: {params['learning_rate']:.3f} | "
                  f"Seed: {params['seed']:3d} | "
                  f"Actual Hidden: {len(network.hidden_units):1d} | "
                  f"Time: {training_time:.2f}s")
            
            results.append({
                'config_id': i,
                'final_loss': final_loss,
                'accuracy': accuracy,
                'accuracy_error': accuracy_error,
                'max_hidden_units': params['max_hidden_units'],
                'use_bias': params['use_bias'],
                'weight_init_range': params['weight_init_range'],
                'learning_rate': params['learning_rate'],
                'seed': params['seed'],
                'training_time': training_time,
                'actual_hidden_units': len(network.hidden_units)
            })
            
        except Exception as e:
            print(f"Config {i:3d}/{len(combinations)} | ERROR: {str(e)}")
            continue
    
    if not results:
        print("No successful configurations found!")
        return [], {}
    
    # Trova la migliore configurazione
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "=" * 120)
    print("BEST CONFIGURATION:")
    print(f"Config {best_result['config_id']} | "
          f"Loss: {best_result['final_loss']:.6f} | "
          f"Accuracy: {best_result['accuracy']:.4f} ± {best_result['accuracy_error']:.4f}")
    print(f"Parameters: Hidden={best_result['max_hidden_units']}, "
          f"Bias={best_result['use_bias']}, "
          f"Weight_Init={best_result['weight_init_range']}, "
          f"LR={best_result['learning_rate']}, "
          f"Seed={best_result['seed']}")
    
    return results, best_result

def analyze_results(results):
    """Analizza i risultati della grid search"""
    if not results:
        print("No results to analyze!")
        return None
        
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS BY PARAMETER:")
    print("=" * 80)
    
    # Analisi per numero di hidden units
    print("\n1. PERFORMANCE BY HIDDEN UNITS:")
    hidden_stats = results_df.groupby('max_hidden_units').agg({
        'accuracy': ['mean', 'std', 'max'],
        'final_loss': ['mean', 'min']
    }).round(4)
    print(hidden_stats)
    
    # Analisi per bias
    print("\n2. PERFORMANCE BY BIAS USAGE:")
    bias_stats = results_df.groupby('use_bias').agg({
        'accuracy': ['mean', 'std', 'max'],
        'final_loss': ['mean', 'min']
    }).round(4)
    print(bias_stats)
    
    # Analisi per learning rate
    print("\n3. PERFORMANCE BY LEARNING RATE:")
    lr_stats = results_df.groupby('learning_rate').agg({
        'accuracy': ['mean', 'std', 'max'],
        'final_loss': ['mean', 'min']
    }).round(4)
    print(lr_stats)
    
    # Top 5 configurazioni
    print("\n4. TOP 5 CONFIGURATIONS:")
    top_5 = results_df.nlargest(5, 'accuracy')[[
        'config_id', 'accuracy', 'accuracy_error', 'final_loss', 
        'max_hidden_units', 'use_bias', 'learning_rate', 'seed'
    ]]
    print(top_5.to_string(index=False))
    
    return results_df

def main():
    # Esegui grid search
    print("MONK-1 Cascade Correlation Grid Search")
    print("Hidden Units: 3 to 6")
    results, best_result = grid_search_monk1()
    
    if results:
        # Analizza i risultati
        results_df = analyze_results(results)
        
        # Salva i risultati
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"monk1_grid_search_results_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        return results_df, best_result
    else:
        print("No results to return!")
        return None, None

if __name__ == "__main__":
    results_df, best_result = main()