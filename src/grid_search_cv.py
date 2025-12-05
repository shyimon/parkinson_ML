import numpy as np
import time
import os
import json
import itertools

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
        
        # range iniziali per ciascun parametro
        self.lr_range = [0.01, 0.25]
        self.hidden_range = [2, 6]
        self.epochs_range = [300, 600]
    
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
    
    def _evaluate_params(self, params, X, y):
        folds = self._create_folds(X, y, seed=params.get('cv_seed', 42))
        accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            try:
                acc = self._train_single_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, params)
                accuracies.append(acc)
                if self.verbose:
                    print(f"    Fold {fold+1}: {acc:.1f}%", end='  ')
            except Exception as e:
                accuracies.append(0.0)
                if self.verbose:
                    print(f"    Fold {fold+1}: errore", end='  ')
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            if self.verbose:
                print(f"\n    Media: {mean_acc:.1f}% ± {std_acc:.1f}%")
            return mean_acc, std_acc
        
        return 0.0, 0.0
    
    def _train_single_model(self, X_train, y_train, X_val, y_val, params):
        import neural_network as nn

        input_size = X_train.shape[1]
        structure = [input_size, params['hidden_neurons'], 1]
        net = nn.NeuralNetwork(structure, eta=params['learning_rate'])

        # Inizializzazione Xavier per tanh
        for layer_idx, layer in enumerate(net.layers):
            if layer_idx == 0:
                continue
            for neuron in layer:
                n_inputs = len(neuron.weights)
                if n_inputs > 0:
                    limit = np.sqrt(6.0 / (n_inputs + 1))
                    neuron.weights = np.random.uniform(-limit, limit, n_inputs)
                neuron.bias = 0.0
        
        net.fit(X_train, X_val, y_train, y_val, epochs=params['epochs'])
        predictions = net.predict(X_val)
        pred_classes = (predictions >= 0.5).astype(int)
        return np.mean(pred_classes == y_val) * 100
    
    def _grid_points(self, ranges_dict, n_points=3):
        # genero punti della griglia uniformemente distribuiti nei range
        grid_points = []

        # per ogni parametro genero n_points valori uniformemente distribuiti
        param_values = {}
        for param_name, (min_val, max_val) in ranges_dict.items():
            if param_name in ['hidden_neurons', 'epochs']:  # Parametri discreti
                values = np.linspace(min_val, max_val, n_points).astype(int)
                values = np.unique(values)  # Rimuove i duplicati
            else:  # Parametri continui
                values = np.linspace(min_val, max_val, n_points)
            param_values[param_name] = values.tolist()
    
         # Prodotto cartesiano di tutti i valori
        param_names = list(param_values.keys())
        value_combinations = list(itertools.product(*[param_values[name] for name in param_names]))

         # Converti in dizionari
        for combo in value_combinations:
            point = {param_names[i]: combo[i] for i in range(len(param_names))}
            grid_points.append(point)
        
        return grid_points
    
    def _dichotomic_search(self, X, y, max_iteration=3, n_points=3):

        print("\n" + "="*60)
        print("RICERCA DICOTOMICA N-DIMENSIONALE")
        print("="*60)

        # range iniziali
        current_ranges = {'learning_rate': tuple(self.lr_range), 
                          'hidden_neurons': tuple(self.hidden_range),
                          'epochs': tuple(self.epochs_range)
                          }
        best_overall = {'params': None, 'mean': -np.inf, 'std': 0}

    # ogni iterazione genera una griglia di n_points per parametro
    # restringe i range del 50% attorno al punto migliore
    # esplora tutte le combinazioni contemporaneamente
    # funzione che aluta tutte le combinazioni di parametri contemporaneamente.
        for iteration in range(max_iteration):
            print(f"\n{'='*60}")
            print(f"ITERAZIONE {iteration + 1}")
            print(f"{'='*60}")

            print(f"Range attuali:")
            for param, (min_val, max_val) in current_ranges.items():
                print(f"  {param}: [{min_val}, {max_val}]")

            # Genera punti della griglia
            grid_points = self._grid_points(current_ranges, n_points)
            print(f"Punti da testare: {len(grid_points)}")
            
            # Valuta tutti i punti
            iteration_best = {'params': None, 'mean': -np.inf, 'std': 0}

            for i, params in enumerate(grid_points):
                params['cv_seed'] = 42
                if self.verbose:
                    print(f"\n[{i+1}/{len(grid_points)}] ", end="")
                
                mean_acc, std_acc = self._evaluate_params(params, X, y)
                
                self.results.append({
                    'params': params.copy(),
                    'mean_accuracy': mean_acc,
                    'iteration': iteration
                })
                
                if mean_acc > iteration_best['mean']:
                    iteration_best = {'params': params.copy(), 'mean': mean_acc, 'std': std_acc}
                    if self.verbose:
                        print("  → Migliore!")
            
            print(f"\nMiglior combinazione iterazione {iteration + 1}:")
            for param, value in iteration_best['params'].items():
                if param != 'cv_seed':
                    print(f"  {param}: {value}")
            print(f"  Accuracy: {iteration_best['mean']:.1f}%")

             # Aggiorna migliore globale
            if iteration_best['mean'] > best_overall['mean']:
                best_overall = iteration_best.copy()
                print("  → Nuovo migliore globale!")
            
            # Restringo i range attorno al punto migliore
            new_ranges = {}
            for param, (min_val, max_val) in current_ranges.items():
                best_val = iteration_best['params'][param]
                
                # Calcolo nuovo range (restringe del 50% ogni iterazione)
                range_width = max_val - min_val
                new_min = max(best_val - range_width/4, min_val)
                new_max = min(best_val + range_width/4, max_val)
                
                # Assicura che il range non diventi troppo piccolo
                if param in ['hidden_neurons', 'epochs']:  # Parametri discreti
                    if new_max - new_min < 2:
                        new_min = max(best_val - 1, min_val)
                        new_max = min(best_val + 1, max_val)
                else:  # Parametri continui
                    if new_max - new_min < 0.01:
                        new_min = max(best_val - 0.01, min_val)
                        new_max = min(best_val + 0.01, max_val)
                
                new_ranges[param] = (new_min, new_max)
            
            current_ranges = new_ranges
        
        # Salva risultati finali
        self.best_params = best_overall['params'].copy()
        if 'cv_seed' in self.best_params:
            del self.best_params['cv_seed']
        
        self.best_mean = best_overall['mean']
        self.best_std = best_overall['std']
        
        self._print_results()
        self._save_results()
        
        return self.best_params, self.best_mean, self.best_std

    def _print_results(self):
        print("\n" + "="*60)
        print("RISULTATI FINALI")
        print("="*60)
        
        if self.best_params:
            print(f"\nPARAMETRI OTTIMALI:")
            for param, value in self.best_params.items():
                print(f"  {param:15s}: {value}")
            print(f"\n  Accuracy: {self.best_mean:.1f}% ± {self.best_std:.1f}%")
            print(f"  Test totali: {len(self.results)}")
    
    def _save_results(self):
        if not self.best_params:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"nd_dichotomic_{timestamp}.json")
        
        data_to_save = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'best_params': self.best_params,
            'cv_accuracy': float(self.best_mean),
            'cv_std': float(self.best_std),
            'search_type': 'nd_dichotomic',
            'initial_ranges': {
                'learning_rate': self.lr_range,
                'hidden_neurons': self.hidden_range,
                'epochs': self.epochs_range
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"\nRisultati salvati in: {filepath}")