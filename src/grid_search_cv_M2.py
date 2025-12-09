import numpy as np
import time
import os
import json
import itertools

class GridSearchM2:
    def __init__(self, cv_folds=5, verbose=True, results_dir='grid_search_results_monk2'):
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = []
        self.best_params = None
        self.best_mean = -np.inf
        self.best_std = 0
        
        # Range ottimizzati per MONK2 (
        self.lr_range = [0.001, 0.1]     
        self.hidden_range = [3, 10]       
        self.epochs_range = [400, 1000]   
        self.batch_range = [16, 100]

    def _create_folds(self, X, y, seed=42):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        
        fold_size = n_samples // self.cv_folds
        extra_samples = n_samples % self.cv_folds
        
        folds = []
        current_start = 0
        
        for i in range(self.cv_folds):
            current_fold_size = fold_size + (1 if i < extra_samples else 0)
            start = current_start
            end = current_start + current_fold_size
            
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_indices, val_indices))
            
            current_start = end
        
        return folds
    
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
        
        batch_size = params.get('batch_size', 1)
        net.fit(X_train, X_val, y_train, y_val, epochs=params['epochs'], batch_size=batch_size)
        predictions = net.predict(X_val)
        pred_classes = (predictions >= 0.5).astype(int)
        return np.mean(pred_classes == y_val) * 100
    
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
    
    def _grid_points(self, ranges_dict, n_points=5):
        grid_points = []
        # per ogni parametro genero n_points valori uniformemente distribuiti
        param_values = {}
        
        for param_name, (min_val, max_val) in ranges_dict.items():
            if param_name in ['hidden_neurons', 'epochs', 'batch size']:
                values = np.linspace(min_val, max_val, n_points).astype(int)
                values = np.unique(values)
                values = np.linspace(min_val, max_val, n_points).astype(int)
                values = np.unique(values)  # Rimuove i duplicati
                # Per batch_size, arrotonda alle potenze di 2 più vicine
                if param_name == 'batch_size':
                    values = [self._nearest_power_of_two(v) for v in values]
            else:  # Parametri continui
                values = np.linspace(min_val, max_val, n_points)
            param_values[param_name] = values.tolist()
        
        param_names = list(param_values.keys())
        value_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        for combo in value_combinations:
            point = {param_names[i]: combo[i] for i in range(len(param_names))}
            grid_points.append(point)
        
        return grid_points
    
    def _nearest_power_of_two(self, n):
        """Trova la potenza di 2 più vicina a n"""
        if n <= 1:
            return 1
        powers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        return min(powers, key=lambda x: abs(x - n))

    def _local_refinement(self, X, y, best_params, radius=0.1, n_points=5):
        print(f"\nRaffinamento attorno a:")
        for param, value in best_params.items():
            if param != 'cv_seed':
                print(f"  {param}: {value}")
        
        refined_ranges = {}
        
        # Definisci range ristretti attorno ai migliori parametri
        for param, value in best_params.items():
            if param == 'cv_seed':
                continue
                
            if param == 'learning_rate':
                min_val = max(value * (1 - radius), 0.0005)
                max_val = min(value * (1 + radius), 0.2)
            elif param == 'hidden_neurons':
                min_val = max(value - 3, 2)
                max_val = value + 3
            elif param == 'epochs':
                min_val = max(value - 100, 200)
                max_val = value + 100
            elif param == 'batch_size':
                # Per batch_size, cerca potenze di 2 vicine
                current_powers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
                idx = current_powers.index(value) if value in current_powers else 0
                min_val = current_powers[max(idx - 1, 0)]
                max_val = current_powers[min(idx + 1, len(current_powers) - 1)]
            else:
                min_val = value * 0.9
                max_val = value * 1.1
            
            refined_ranges[param] = (min_val, max_val)
        
        # Crea griglia fine
        grid_points = self._grid_points(refined_ranges, n_points)
        print(f"Punti di raffinamento: {len(grid_points)}")
        
        # Valuta tutti i punti
        best_refined = {'params': None, 'mean': -np.inf, 'std': 0}
        
        for i, params in enumerate(grid_points):
            params['cv_seed'] = 99  # Seed fisso per raffinamento
            if self.verbose:
                print(f"\nRaffinamento [{i+1}/{len(grid_points)}] ", end="")
            
            mean_acc, std_acc = self._evaluate_params(params, X, y)
            
            self.results.append({
                'params': params.copy(),
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'iteration': 'refinement'
            })
            
            if mean_acc > best_refined['mean']:
                best_refined = {'params': params.copy(), 'mean': mean_acc, 'std': std_acc}
                if self.verbose:
                    print(f"→ Migliore! ({mean_acc:.2f}%)")
        
        return best_refined['params'], best_refined['mean'], best_refined['std']
    
    def _dichotomic_search(self, X, y, max_iteration=4, n_points=5):
        print("\n" + "="*60)
        print("RICERCA DICOTOMICA N-DIMENSIONALE - MONK2")
        print("="*60)

        current_ranges = {
            'learning_rate': tuple(self.lr_range), 
            'hidden_neurons': tuple(self.hidden_range),
            'epochs': tuple(self.epochs_range),
            'batch_size': (min(self.batch_range), max(self.batch_range))
        }
        
        best_overall = {'params': None, 'mean': -np.inf, 'std': 0}

        for iteration in range(max_iteration):
            print(f"\n{'='*60}")
            print(f"ITERAZIONE {iteration + 1}/{max_iteration}")
            print(f"{'='*60}")

            print(f"Range attuali:")
            for param, (min_val, max_val) in current_ranges.items():
                if param == 'learning_rate':
                    print(f"  {param}: [{min_val:.4f}, {max_val:.4f}]")
                else:
                    print(f"  {param}: [{min_val}, {max_val}]")

            grid_points = self._grid_points(current_ranges, n_points)
            print(f"Punti da testare: {len(grid_points)}")
            
            iteration_best = {'params': None, 'mean': -np.inf, 'std': 0}

            for i, params in enumerate(grid_points):
                params['cv_seed'] = 42 + iteration
                if self.verbose:
                    print(f"\n[{i+1}/{len(grid_points)}] ", end="")
                    for p, v in params.items():
                        if p != 'cv_seed':
                            if p == 'learning_rate':
                                print(f"{p}: {v:.4f} ", end="")
                            else:
                                print(f"{p}: {v} ", end="")
                
                mean_acc, std_acc = self._evaluate_params(params, X, y)
                
                self.results.append({
                    'params': params.copy(),
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'iteration': iteration
                })
                
                if mean_acc > iteration_best['mean']:
                    iteration_best = {'params': params.copy(), 'mean': mean_acc, 'std': std_acc}
                    if self.verbose:
                        print(f"  → Migliore! ({mean_acc:.1f}%)")
            
            print(f"\nMiglior combinazione iterazione {iteration + 1}:")
            for param, value in iteration_best['params'].items():
                if param != 'cv_seed':
                    if param == 'learning_rate':
                        print(f"  {param}: {value:.4f}")
                    else:
                        print(f"  {param}: {value}")
            print(f"  Accuracy: {iteration_best['mean']:.2f}% ± {iteration_best['std']:.2f}%")

            if iteration_best['mean'] > best_overall['mean']:
                best_overall = iteration_best.copy()
                print("  → Nuovo migliore globale!")
            
            # Restringimento conservativo del 30%
            new_ranges = {}
            for param, (min_val, max_val) in current_ranges.items():
                best_val = iteration_best['params'][param]
                
                range_width = max_val - min_val
                new_min = max(best_val - range_width * 0.15, min_val)
                new_max = min(best_val + range_width * 0.15, max_val)
                
                if param in ['hidden_neurons', 'epochs']:
                    if new_max - new_min < 3:
                        new_min = max(best_val - 2, min_val)
                        new_max = min(best_val + 2, max_val)
                else:
                    if new_max - new_min < 0.005:
                        new_min = max(best_val - 0.005, min_val)
                        new_max = min(best_val + 0.005, max_val)
                
                new_ranges[param] = (new_min, new_max)
            
            current_ranges = new_ranges
        
        # Fase di raffinamento finale
        print("\n" + "="*60)
        print("RAFFINAMENTO FINALE")
        print("="*60)
        
        refined_params, refined_mean, refined_std = self._local_refinement(
            X, y, best_overall['params'], radius=0.1, n_points=5
        )
        
        if refined_mean > best_overall['mean']:
            best_overall = {
                'params': refined_params,
                'mean': refined_mean,
                'std': refined_std
            }
            print("  → Parametri raffinati migliori dei precedenti!")
        
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
        print("RISULTATI FINALI - MONK2")
        print("="*60)
        
        if self.best_params:
            print(f"\nPARAMETRI OTTIMALI:")
            for param, value in self.best_params.items():
                if param == 'learning_rate':
                    print(f"  {param:15s}: {value:.4f}")
                else:
                    print(f"  {param:15s}: {value}")
            print(f"\n  Accuracy: {self.best_mean:.2f}% ± {self.best_std:.2f}%")
            print(f"  Test totali: {len(self.results)}")
    
    def _save_results(self):
        if not self.best_params:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"monk2_dichotomic_{timestamp}.json")
        
        data_to_save = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'best_params': self.best_params,
            'cv_accuracy': float(self.best_mean),
            'cv_std': float(self.best_std),
            'search_type': 'monk2_dichotomic',
            'initial_ranges': {
                'learning_rate': self.lr_range,
                'hidden_neurons': self.hidden_range,
                'epochs': self.epochs_range
            },
            'total_tests': len(self.results)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"\nRisultati salvati in: {filepath}")
    
    def run_trials(self, X_train, y_train, X_test, y_test, n_trials=500, 
                   save_results=True, results_filename='trials_results_monk2.json'):
        """
        Esegue n_trials con i parametri ottimali trovati per MONK2
        """
        if self.best_params is None:
            print("Errore: Nessun parametro ottimale trovato. Esegui prima la ricerca dicotomica.")
            return None
        
        print("\n" + "="*60)
        print(f"ESECUZIONE DI {n_trials} TRIAL - MONK2")
        print("="*60)
        
        import neural_network as nn
        
        accuracies = []
        times = []
        
        print(f"Parametri usati per {n_trials} trial:")
        for param, value in self.best_params.items():
            if param == 'learning_rate':
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")
        
        print(f"\nInizio {n_trials} addestramenti...")
        
        for i in range(n_trials):
            trial_start_time = time.time()
            
            if (i + 1) % 5 == 0:
                print(f"  Trial {i+1}/{n_trials}...")
            
            # Crea rete con struttura dai parametri ottimali
            input_size = X_train.shape[1]
            structure = [input_size, self.best_params['hidden_neurons'], 1]
            net = nn.NeuralNetwork(structure, eta=self.best_params['learning_rate'])
            
            # Imposto tanh e inizializzazione Xavier
            for layer_idx, layer in enumerate(net.layers):
                if layer_idx == 0:
                    continue
                for neuron in layer:
                    neuron.activation_function_type = "tanh"
                    
                    n_inputs = len(neuron.weights)
                    if n_inputs > 0:
                        limit = np.sqrt(6.0 / (n_inputs + 1))
                        neuron.weights = np.random.uniform(-limit, limit, n_inputs)
                    neuron.bias = 0.0
            
            # Allena la rete
            batch_size = self.best_params.get('batch_size', 1)
            net.fit(X_train, X_test, y_train, y_test, 
                    epochs=self.best_params['epochs'], batch_size=batch_size, verbose=False)
            
            # Valuta sul test set
            predictions = net.predict(X_test)
            pred_classes = (predictions >= 0.5).astype(int)
            accuracy = np.mean(pred_classes == y_test) * 100
            accuracies.append(accuracy)
            
            trial_time = time.time() - trial_start_time
            times.append(trial_time)
        
        # Calcola statistiche
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        total_time = np.sum(times)
        
        print(f"\nStatistiche su {n_trials} trial:")
        print("-"*60)
        print(f"Tempo totale: {total_time:.1f} secondi")
        print(f"Tempo medio per trial: {np.mean(times):.1f} secondi")
        print(f"\nAccuracy media: {mean_acc:.2f}%")
        print(f"Deviazione standard: {std_acc:.2f}%")

        
        # Salva risultati dei trial
        if save_results:
            trials_data = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'n_trials': n_trials,
                'best_params': self.best_params,
                'accuracies': accuracies,
                'times': times,
                'statistics': {
                    'mean_accuracy': float(mean_acc),
                    'std_accuracy': float(std_acc),
                    'min_accuracy': float(min_acc),
                    'max_accuracy': float(max_acc),
                    'total_time': float(total_time),
                    'avg_time_per_trial': float(np.mean(times))
                }
            }
            
            filepath = os.path.join(self.results_dir, results_filename)
            with open(filepath, 'w') as f:
                json.dump(trials_data, f, indent=2)
            
            print(f"\nRisultati dei trial salvati in: {filepath}")
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc,
            'accuracies': accuracies,
            'total_time': total_time
        }