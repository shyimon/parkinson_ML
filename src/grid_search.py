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
        self.lr_range = [0.001, 0.6]
        self.hidden_range = [2, 6]
        self.epochs_range = [1000, 2000]
        '''self.hidden_layers_range = [[2], [3], [4], [6], [8],
                              [2, 2], [4, 2], [6, 4], [6, 2], [8, 4], [8, 6],
                              [6, 4, 2], [8, 6, 4], [8, 6, 2],
                              [8, 6, 4, 2]]'''
        # self.l2_reg_range = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
        # da aggiungere weight initializer
        # da aggiungere activation type
        # da aggiungere loss type
        # da aggiungere algorithm
        # self.batch_size_range = [1 << i for i in range(X_train_size.bit_length())]
        # self.batch_size_range.append(X_train_size)
    
    def _create_folds(self, X, seed=42):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # shuffle con seed fisso per riproducibilità
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        
        # calcola dimensione di ogni fold
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
                
            # aggiorna il prossimo fold
            current_start = end

        return folds
    
    def _train_single_model(self, X_train, y_train, X_val, y_val, params):
        import neural_network as nn

        structure = [X_train.shape[1], params['hidden_neurons'], y_train.shape[1]]
        net = nn.NeuralNetwork(structure, eta=params['learning_rate'])

        net.fit(X_train, y_train, X_val, y_val, epochs=params['epochs'])
        
        # Calcola accuracy sia su training che validation
        train_predictions = net.predict(X_train)
        train_pred_classes = (train_predictions >= 0.5).astype(int)
        train_acc = np.mean(train_pred_classes == y_train) * 100
        
        val_predictions = net.predict(X_val)
        val_pred_classes = (val_predictions >= 0.5).astype(int)
        val_acc = np.mean(val_pred_classes == y_val) * 100
        
        return train_acc, val_acc
    
    def _evaluate_params(self, params, X, y):
        folds = self._create_folds(X, y, seed=params.get('cv_seed', 42))
        train_accuracies = []
        val_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            try:
                train_acc, val_acc = self._train_single_model(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold, params)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                if self.verbose:
                    print(f"    Fold {fold+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%", end='  ')
            except Exception as e:
                train_accuracies.append(0.0)
                val_accuracies.append(0.0)
                if self.verbose:
                    print(f"    Fold {fold+1}: errore", end='  ')
        
        if train_accuracies and val_accuracies:
            mean_train_acc = np.mean(train_accuracies)
            std_train_acc = np.std(train_accuracies)
            mean_val_acc = np.mean(val_accuracies)
            std_val_acc = np.std(val_accuracies)
            
            if self.verbose:
                print(f"\n    Training: {mean_train_acc:.1f}% ± {std_train_acc:.1f}%")
                print(f"    Validation: {mean_val_acc:.1f}% ± {std_val_acc:.1f}%")
            
            return mean_val_acc, std_val_acc, mean_train_acc, std_train_acc
        
        return 0.0, 0.0, 0.0, 0.0
    
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
    
    def _local_refinement(self, X, y, best_params, radius=0.1, n_points=5):
        # fase di raffinamento locale attorno ai parametri migliori
        print(f"\nRaffinamento attorno a:")
        for param, value in best_params.items():
            if param != 'cv_seed':
                print(f"{param}: {value}")
        refined_ranges = {}

        # definisco range ristretti attorno ai migliori parametri
        for param, value in best_params.items():
            if param == 'cv_seed':
                continue

            if param == 'learning_rate':
                min_val = max(value * (1 - radius), 0.001)
                max_val = min(value * (1 + radius), 1.0)
            elif param == 'hidden_neurons':
                min_val = max(value - 2, 1)
                max_val = value + 2
            elif param == 'epochs':
                min_val = max(value - 50, 100)
                max_val = value + 50
            else:
                min_val = value * 0.9
                max_val = value * 1.1
            
            refined_ranges[param] = (min_val, max_val)
        
         # Crea griglia fine
        grid_points = self._grid_points(refined_ranges, n_points)
        print(f"Punti di raffinamento: {len(grid_points)}")
        
        # Valuta tutti i punti
        best_refined = {'params': None, 'val_mean': -np.inf, 'val_std': 0, 
                       'train_mean': 0, 'train_std': 0}
        
        for i, params in enumerate(grid_points):
            params['cv_seed'] = 99  # Seed fisso per raffinamento
            if self.verbose:
                print(f"\nRaffinamento [{i+1}/{len(grid_points)}] ", end="")
            
            val_mean, val_std, train_mean, train_std = self._evaluate_params(params, X, y)
            
            self.results.append({
                'params': params.copy(),
                'validation_mean_accuracy': val_mean,
                'validation_std_accuracy': val_std,
                'training_mean_accuracy': train_mean,
                'training_std_accuracy': train_std,
                'iteration': 'refinement'
            })
            
            if val_mean > best_refined['val_mean']:
                best_refined = {
                    'params': params.copy(), 
                    'val_mean': val_mean, 
                    'val_std': val_std,
                    'train_mean': train_mean,
                    'train_std': train_std
                }
                if self.verbose:
                    print(f"→ Migliore! (Val: {val_mean:.2f}%, Train: {train_mean:.2f}%)")
        
        return (best_refined['params'], best_refined['val_mean'], 
                best_refined['val_std'], best_refined['train_mean'], best_refined['train_std'])

    def _dichotomic_search(self, X, y, max_iteration=3, n_points=3):
        print("\n" + "="*60)
        print("RICERCA DICOTOMICA N-DIMENSIONALE")
        print("="*60)

        # range iniziali
        current_ranges = {'learning_rate': tuple(self.lr_range), 
                          'hidden_neurons': tuple(self.hidden_range),
                          'epochs': tuple(self.epochs_range)
                          }
        best_overall = {'params': None, 'val_mean': -np.inf, 'val_std': 0,
                       'train_mean': 0, 'train_std': 0}

        for iteration in range(max_iteration):
            print(f"\n{'='*60}")
            print(f"ITERAZIONE {iteration + 1}")
            print(f"{'='*60}")

            print(f"Range attuali:")
            for param, (min_val, max_val) in current_ranges.items():
                print(f"  {param}: [{min_val:.4f}, {max_val:.4f}]")

            # Genera punti della griglia
            grid_points = self._grid_points(current_ranges, n_points)
            print(f"Punti da testare: {len(grid_points)}")
            
            # Valuta tutti i punti
            iteration_best = {'params': None, 'val_mean': -np.inf, 'val_std': 0,
                             'train_mean': 0, 'train_std': 0}

            for i, params in enumerate(grid_points):
                # seed diverso per ogni iterazione
                params['cv_seed'] = 42 + iteration
                if self.verbose:
                    print(f"\n[{i+1}/{len(grid_points)}] ", end="")
                    for p, v in params.items():
                        if p != 'cv_seed':
                            print(f"{p}: {v} ", end="")
                
                val_mean, val_std, train_mean, train_std = self._evaluate_params(params, X, y)
                
                self.results.append({
                    'params': params.copy(),
                    'validation_mean_accuracy': val_mean,
                    'validation_std_accuracy': val_std,
                    'training_mean_accuracy': train_mean,
                    'training_std_accuracy': train_std,
                    'iteration': iteration
                })
                
                if val_mean > iteration_best['val_mean']:
                    iteration_best = {
                        'params': params.copy(), 
                        'val_mean': val_mean, 
                        'val_std': val_std,
                        'train_mean': train_mean,
                        'train_std': train_std
                    }
                    if self.verbose:
                        print(f"  → Migliore! (Val: {val_mean:.1f}%, Train: {train_mean:.1f}%)")
            
            print(f"\nMiglior combinazione iterazione {iteration + 1}:")
            for param, value in iteration_best['params'].items():
                if param != 'cv_seed':
                    print(f"  {param}: {value}")
            print(f"  Validation Accuracy: {iteration_best['val_mean']:.2f}% ± {iteration_best['val_std']:.2f}%")
            print(f"  Training Accuracy: {iteration_best['train_mean']:.2f}% ± {iteration_best['train_std']:.2f}%")

             # Aggiorna migliore globale
            if iteration_best['val_mean'] > best_overall['val_mean']:
                best_overall = iteration_best.copy()
                print("  → Nuovo migliore globale!")
            
            # Restringo i range attorno al punto migliore (del 30%)
            new_ranges = {}
            for param, (min_val, max_val) in current_ranges.items():
                best_val = iteration_best['params'][param]
                
                # Calcolo nuovo range (restringe del 50% ogni iterazione)
                range_width = max_val - min_val
                new_min = max(best_val - range_width*0.15, min_val)
                new_max = min(best_val + range_width*0.15, max_val)
                
                # Assicura che il range non diventi troppo piccolo
                if param in ['hidden_neurons', 'epochs']:  
                    if new_max - new_min < 2:
                        new_min = max(best_val - 2, min_val)
                        new_max = min(best_val + 2, max_val)
                else:  # Parametri continui
                    if new_max - new_min < 0.01:
                        new_min = max(best_val - 0.01, min_val)
                        new_max = min(best_val + 0.01, max_val)
                
                new_ranges[param] = (new_min, new_max)
            
            current_ranges = new_ranges
        
        # Fase di raffinamento finale
        print("\n" + "="*60)
        print("RAFFINAMENTO FINALE")
        print("="*60)
        
        (refined_params, refined_val_mean, refined_val_std, 
         refined_train_mean, refined_train_std) = self._local_refinement(
            X, y, best_overall['params'], radius=0.1, n_points=5
        )
        
        if refined_val_mean > best_overall['val_mean']:
            best_overall = {
                'params': refined_params,
                'val_mean': refined_val_mean,
                'val_std': refined_val_std,
                'train_mean': refined_train_mean,
                'train_std': refined_train_std
            }
            print("  → Parametri raffinati migliori dei precedenti!")

        # Salva risultati finali
        self.best_params = best_overall['params'].copy()
        if 'cv_seed' in self.best_params:
            del self.best_params['cv_seed']
        
        self.best_mean = best_overall['val_mean']
        self.best_std = best_overall['val_std']
        self.best_train_mean = best_overall['train_mean']
        self.best_train_std = best_overall['train_std']
        
        self._print_results()
        self._save_results()
        
        return (self.best_params, self.best_mean, self.best_std, 
                self.best_train_mean, self.best_train_std)

    def _print_results(self):
        print("\n" + "="*60)
        print("RISULTATI FINALI")
        print("="*60)
        
        if self.best_params:
            print(f"\nPARAMETRI OTTIMALI:")
            for param, value in self.best_params.items():
                print(f"  {param:15s}: {value}")
            print(f"\n  Training Accuracy: {self.best_train_mean:.2f}% ± {self.best_train_std:.2f}%")
            print(f"  Validation Accuracy: {self.best_mean:.2f}% ± {self.best_std:.2f}%")
            print(f"  Gap (Train-Val): {self.best_train_mean - self.best_mean:.2f}%")
            print(f"  Test totali: {len(self.results)}")
    
    def _save_results(self):
        if not self.best_params:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"nd_dichotomic_{timestamp}.json")
        
        data_to_save = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'best_params': self.best_params,
            'training_accuracy': float(self.best_train_mean),
            'training_std': float(self.best_train_std),
            'validation_accuracy': float(self.best_mean),
            'validation_std': float(self.best_std),
            'accuracy_gap': float(self.best_train_mean - self.best_mean),
            'search_type': 'nd_dichotomic',
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
                   save_results=True, results_filename='trials_results.json'):
        """
        Esegue n_trials con i parametri ottimali trovati
        Calcola training accuracy e test accuracy con le loro deviazioni standard
        """
        if self.best_params is None:
            print("Errore: Nessun parametro ottimale trovato. Esegui prima la ricerca dicotomica.")
            return None
        
        print("\n" + "="*60)
        print(f"ESECUZIONE DI {n_trials} TRIAL")
        print("="*60)
        
        import neural_network as nn
        
        train_accuracies = []
        test_accuracies = []
        times = []
        
        print(f"Parametri usati per {n_trials} trial:")
        for param, value in self.best_params.items():
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
                if layer_idx == 0:  # Input layer 
                    continue
                for neuron in layer:
                    # Cambia a tanh
                    neuron.activation_function_type = "tanh"
                    
                    # Inizializzazione Xavier per tanh
                    n_inputs = len(neuron.weights)
                    if n_inputs > 0:
                        limit = np.sqrt(6.0 / (n_inputs + 1))
                        neuron.weights = np.random.uniform(-limit, limit, n_inputs)
                    neuron.bias = 0.0
            
            # Allena la rete
            net.fit(X_train, X_test, y_train, y_test, 
                    epochs=self.best_params['epochs'], verbose=False)
            
            # Calcola training accuracy
            train_predictions = net.predict(X_train)
            train_pred_classes = (train_predictions >= 0.5).astype(int)
            train_accuracy = np.mean(train_pred_classes == y_train) * 100
            train_accuracies.append(train_accuracy)
            
            # Calcola test accuracy
            test_predictions = net.predict(X_test)
            test_pred_classes = (test_predictions >= 0.5).astype(int)
            test_accuracy = np.mean(test_pred_classes == y_test) * 100
            test_accuracies.append(test_accuracy)
            
            # Tempo del trial
            trial_time = time.time() - trial_start_time
            times.append(trial_time)
        
        # Calcola statistiche per training accuracy
        mean_train_acc = np.mean(train_accuracies)
        std_train_acc = np.std(train_accuracies)
        min_train_acc = np.min(train_accuracies)
        max_train_acc = np.max(train_accuracies)
        
        # Calcola statistiche per test accuracy
        mean_test_acc = np.mean(test_accuracies)
        std_test_acc = np.std(test_accuracies)
        min_test_acc = np.min(test_accuracies)
        max_test_acc = np.max(test_accuracies)
        
        # Calcola gap medio tra training e test
        accuracy_gaps = [train_acc - test_acc for train_acc, test_acc in zip(train_accuracies, test_accuracies)]
        mean_gap = np.mean(accuracy_gaps)
        std_gap = np.std(accuracy_gaps)
        
        total_time = np.sum(times)
        
        print(f"\n{'='*60}")
        print(f"STATISTICHE SU {n_trials} TRIAL")
        print(f"{'='*60}")
        print(f"Tempo totale: {total_time:.1f} secondi")
        print(f"Tempo medio per trial: {np.mean(times):.1f} secondi")
        
        print(f"\n{'─'*60}")
        print(f"TRAINING ACCURACY:")
        print(f"{'─'*60}")
        print(f"Media: {mean_train_acc:.2f}%")
        print(f"Deviazione standard: {std_train_acc:.2f}%")
        print(f"Minimo: {min_train_acc:.2f}%")
        print(f"Massimo: {max_train_acc:.2f}%")
        print(f"Range: [{mean_train_acc - std_train_acc:.2f}%, {mean_train_acc + std_train_acc:.2f}%]")
        
        print(f"\n{'─'*60}")
        print(f"TEST ACCURACY:")
        print(f"{'─'*60}")
        print(f"Media: {mean_test_acc:.2f}%")
        print(f"Deviazione standard: {std_test_acc:.2f}%")
        print(f"Minimo: {min_test_acc:.2f}%")
        print(f"Massimo: {max_test_acc:.2f}%")
        print(f"Range: [{mean_test_acc - std_test_acc:.2f}%, {mean_test_acc + std_test_acc:.2f}%]")
        
        print(f"\n{'─'*60}")
        print(f"GAP TRAINING-TEST:")
        print(f"{'─'*60}")
        print(f"Gap medio (Train - Test): {mean_gap:.2f}%")
        print(f"Deviazione standard del gap: {std_gap:.2f}%")
        print(f"Range gap: [{mean_gap - std_gap:.2f}%, {mean_gap + std_gap:.2f}%]")
        
        # Salva risultati dei trial
        if save_results:
            trials_data = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'n_trials': n_trials,
                'best_params': self.best_params,
                'training_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
                'accuracy_gaps': accuracy_gaps,
                'times': times,
                'statistics': {
                    'training_accuracy': {
                        'mean': float(mean_train_acc),
                        'std': float(std_train_acc),
                        'min': float(min_train_acc),
                        'max': float(max_train_acc)
                    },
                    'test_accuracy': {
                        'mean': float(mean_test_acc),
                        'std': float(std_test_acc),
                        'min': float(min_test_acc),
                        'max': float(max_test_acc)
                    },
                    'accuracy_gap': {
                        'mean': float(mean_gap),
                        'std': float(std_gap)
                    },
                    'time': {
                        'total': float(total_time),
                        'avg_per_trial': float(np.mean(times))
                    }
                }
            }

            filepath = os.path.join(self.results_dir, results_filename)
            with open(filepath, 'w') as f:
                json.dump(trials_data, f, indent=2)
            
            print(f"\nRisultati dei trial salvati in: {filepath}")
        
        return {
            'training_accuracy': {
                'mean': mean_train_acc,
                'std': std_train_acc,
                'min': min_train_acc,
                'max': max_train_acc,
                'values': train_accuracies
            },
            'test_accuracy': {
                'mean': mean_test_acc,
                'std': std_test_acc,
                'min': min_test_acc,
                'max': max_test_acc,
                'values': test_accuracies
            },
            'accuracy_gap': {
                'mean': mean_gap,
                'std': std_gap,
                'values': accuracy_gaps
            },
            'time': {
                'total': total_time,
                'avg_per_trial': np.mean(times)
            }
        }