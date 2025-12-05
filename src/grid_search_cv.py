import numpy as np
import time
import os
import json

class GridSearch:
    def __init__(self, cv_folds=5, verbose=True, results_dir='grid_search_results'):
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = []
        self.best_params = None
        self.best_mean = -np.inf
        self.best_std = 0
        self.best_fold_accuracies = []
        
        self.lr_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.hidden_range = [2, 3, 4, 6, 8]
        self.epochs_range = [300, 400, 500]
    
    def _create_folds(self, X, y, seed=42):

        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # shuffle con seed fisso per riproducibilità
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        
        #calcola dimensione di ogni fold
        fold_size = n_samples // self.cv_folds

        #calcola i samples rimanenti
        extra_samples = n_samples % self.cv_folds

        folds = []
        current_start = 0
        
        #la riga end = start + fold size if i< self.cv_folds - 1 else n_samples, mette 
        # i rimanenti fold della divisione intera nell'ultimo fold, così l'ultimo fold è 
        # più grande degli altri. Ho fatto un cambiamento per distribuire i samples rimanenti
        # dalla divisione intera tra i vari fold, così i fold hanno un numero di samples
        # più bilanciato
        for i in range(self.cv_folds):
            # Questo fold avrà dimensione fold_size + (1 se i < extra_samples)
            current_fold_size = fold_size + (1 if i < extra_samples else 0)

            start = current_start
            end = current_start + fold_size

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_indices, val_indices))
                
            # aggiorna il prossimo fold
            current_start = end

        return folds
    
    def _binary_search(self, param_name, values, base_params, X, y):
      
        # Ricerca dicotomica per un singolo parametro.
        low = 0
        high = len(values) - 1
        best_value = None
        best_acc = -np.inf
        best_std = 0
        acc_history = []
        
        while low <= high:
            mid = (low + high) // 2

            # Testa il valore centrale
            test_params = base_params.copy()
            test_params[param_name] = values[mid]
            mid_acc, mid_std = self._evaluate_single_params(test_params, X, y)
            acc_history.append((values[mid], mid_acc, mid_std))

            # Testa valori adiacenti se esistono
            if mid > low:
                test_params_low = base_params.copy()
                test_params_low[param_name] = values[mid-1]
                low_acc, low_std = self._evaluate_single_params(test_params_low, X, y)
                acc_history.append((values[mid-1], low_acc, low_std))
            else:
                low_acc = -np.inf
            
            if mid < high:
                test_params_high = base_params.copy()
                test_params_high[param_name] = values[mid+1]
                high_acc, high_std = self._evaluate_single_params(test_params_high, X, y)
                acc_history.append((values[mid+1], high_acc, high_std))
            else:
                high_acc = -np.inf
            
            # Trova il migliore tra i tre
            if mid_acc >= low_acc and mid_acc >= high_acc:
                best_value = values[mid]
                best_acc = mid_acc
                best_std = mid_std
                break
            elif low_acc > mid_acc and low_acc > high_acc:
                high = mid - 1
                if low_acc > best_acc:
                    best_value = values[mid-1]
                    best_acc = low_acc
                    best_std = low_std
            else:
                low = mid + 1
                if high_acc > best_acc:
                    best_value = values[mid+1]
                    best_acc = high_acc
                    best_std = high_std
        
        return best_value, best_acc, best_std
    
    def _evaluate_single_params(self, params, X, y):
       # valuto i parametri su singolo fold di cross-validation
        
        # Calcola input size
        input_size = X.shape[1]

        if self.verbose:
            print(f"\n    Valutazione parametri:")
            print(f"      Learning rate: {params['learning_rate']}")
            print(f"      Hidden neurons: {params['hidden_neurons']}")
            print(f"      Epochs: {params['epochs']}")
            print(f"      Seed: {params.get('cv_seed', 42)}")
        
        # Crea fold
        folds = self._create_folds(X, y, seed=params.get('cv_seed', 42))
        accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            if self.verbose and len(folds) > 1:
                print(f"    Fold {fold+1}/{len(folds)}", end=' ')
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            try:
                # Allena e valuta su questo fold
                acc = self._train_single_model(
                    X_train_fold, y_train_fold, 
                    X_val_fold, y_val_fold, 
                    params, input_size
                )
                accuracies.append(acc)
                if self.verbose and len(folds) > 1:
                    print(f"→ {acc:.2f}%")
            except Exception as e:
                if self.verbose:
                    print(f"Errore nel fold {fold+1}: {e}")
                accuracies.append(0.0)
        
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            if self.verbose:
                print(f"    Media: {mean_acc:.2f}%, Deviazione: ±{std_acc:.2f}%")
            return mean_acc, std_acc
        
        return 0.0, 0.0
    
    def _train_single_model(self, X_train, y_train, X_val, y_val, params, input_size):
        """
        Allena un singolo modello e restituisce accuracy.
        """
        import neural_network as nn
        
        # Creo la rete 
        structure = [input_size, params['hidden_neurons'], 1]
        net = nn.NeuralNetwork(structure, eta=params['learning_rate'])
        
        # Inizializzazione pesi (Xavier per tanh)
        for layer_idx, layer in enumerate(net.layers):
            if layer_idx == 0:  # Input layer - nessun peso
                continue
            
            for neuron in layer:
                # Inizializzazione Xavier per tanh
                n_inputs = len(neuron.weights)
                if n_inputs > 0:
                    limit = np.sqrt(6.0 / (n_inputs + 1))
                    neuron.weights = np.random.uniform(-limit, limit, n_inputs)
                neuron.bias = 0.0  # Bias inizializzato a 0
        
        # Alleno il modello
        net.fit(X_train, X_val, y_train, y_val, 
                epochs=params['epochs'])
        
        # Valutazione
        predictions = net.predict(X_val)
        pred_classes = (predictions >= 0.5).astype(int)
        accuracy = np.mean(pred_classes == y_val) * 100
        
        return accuracy
    
    def search_dichotomic(self, X, y):
       # Ricerca dicotomica per i parametri principali
       
        print("\n" + "="*60)
        print("RICERCA DICOTOMICA DEI PARAMETRI")
        print("="*60)
        
        input_size = X.shape[1]
        
        # Parametri iniziali (valori medi dei range)
        base_params = {
            'learning_rate': self.lr_range[len(self.lr_range)//2],
            'hidden_neurons': self.hidden_range[len(self.hidden_range)//2],
            'epochs': self.epochs_range[len(self.epochs_range)//2],
            'cv_seed': 42
        }
        
        print(f"\nParametri iniziali:")
        for key, value in base_params.items():
            print(f"  {key}: {value}")
        
        # ricerca learning rate
        print(f"\n[FASE 1] Ricerca learning rate ottimale...")
        best_lr, lr_acc, lr_std = self._binary_search(
            'learning_rate', self.lr_range, base_params, X, y
        )
        base_params['learning_rate'] = best_lr
        
        # ricerca neuroni hidden
        print(f"\n[FASE 2] Ricerca neuroni hidden ottimali...")
        best_hidden, hidden_acc, hidden_std = self._binary_search(
            'hidden_neurons', self.hidden_range, base_params, X, y
        )
        base_params['hidden_neurons'] = best_hidden
        
        # ricerca epoche
        print(f"\n[FASE 3] Ricerca epoche ottimali...")
        best_epochs, epochs_acc, epochs_std = self._binary_search(
            'epochs', self.epochs_range, base_params, X, y
        )
        base_params['epochs'] = best_epochs
        
        # Valutazione finale con i migliori parametri
        print(f"\n[VALIDAZIONE FINALE] Test parametri ottimali...")
        final_acc, final_std = self._evaluate_single_params(base_params, X, y)
        
        # Salvo risultati
        self.best_params = base_params.copy()
        self.best_mean = final_acc
        self.best_std = final_std
        
        # Stamp risultati
        self._print_results()
        
        # Salvo su file
        self._save_results()
        
        return self.best_params, self.best_mean, self.best_std
    
    def _print_results(self):
        # stampo i risultati della ricerca dicotomica
        print("\n" + "="*60)
        print("RISULTATI RICERCA DICOTOMICA")
        print("="*60)
        
        if self.best_params:
            print(f"\n PARAMETRI OTTIMALI TROVATI:")
            print(f"  Accuracy: {self.best_mean:.2f}% ± {self.best_std:.2f}%")
            print(f"  Intervallo 95%: [{self.best_mean - 1.96*self.best_std:.2f}%, "
                  f"{self.best_mean + 1.96*self.best_std:.2f}%]")
            
            print(f"\n  Valori dei parametri:")
            for param, value in self.best_params.items():
                print(f"    {param:15s}: {value}")
    
    def _save_results(self):
        # Salva i risultati in formato JSON
        if not self.best_params:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"dichotomic_search_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        data_to_save = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'best_params': self.best_params,
            'cv_accuracy': float(self.best_mean),
            'cv_std': float(self.best_std),
            'search_type': 'dichotomic',
            'parameter_ranges': {
                'learning_rate': self.lr_range,
                'hidden_neurons': self.hidden_range,
                'epochs': self.epochs_range
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"\n Risultati salvati in: {filepath}")

