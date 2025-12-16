import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import neural_network as nn
import data_manipulation as data
from collections import defaultdict
import warnings
import time
warnings.filterwarnings('ignore')

class GridSearch:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = y_test
        self.best_params = None
        self.best_score = -np.inf
        self.search_history = []
        
    def _evaluate_params(self, params, trials=1):
        """Valuta un set di parametri su trials ripetuti"""
        accuracies = []
    
        for trial in range(trials):
            try:
                # Costruisce la struttura della rete
                input_size = self.X_train.shape[1]
                output_size = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1
        
                network_structure = [int(input_size)]  # Converti a int
        
                # Gestione flessibile della struttura
                if 'hidden_structure' in params and params['hidden_structure']:
                    # Usa struttura specificata, convertendo a int
                    network_structure.extend([int(n) for n in params['hidden_structure']])
                else:
                    # Usa numero di layer e neuroni per layer
                    hidden_layers = int(params.get('hidden_layers', 1))
                    hidden_neurons = int(params.get('hidden_neurons', 4))
                    for _ in range(hidden_layers):
                        network_structure.append(int(hidden_neurons))
        
                network_structure.append(int(output_size))  # Converti a int
        
                print(f"  Trial {trial+1}: Struttura rete: {network_structure}")
                print(f"  Input size atteso: {input_size}")

                 #  debug per il primo trial
                debug_mode = (trial == 0)
            
                # Parametri per NeuralNetwork
                nn_params = {'eta': params.get('eta', 0.1),'loss_type': params.get('loss_type', 'half_mse'),
                    'algorithm': params.get('algorithm', 'sgd'),
                    'activation_type': params.get('activation_type', 'sigmoid'),
                    'l2_lambda': params.get('l2_lambda', 0.0),
                    'momentum': params.get('momentum', 0.0),
                    'weight_initializer': params.get('weight_initializer', 'def'),
                    'eta_plus': params.get('eta_plus', 1.2),
                    'eta_minus': params.get('eta_minus', 0.5),
                    'mu': params.get('mu', 1.75),
                    'decay': params.get('decay', 0.0001)
                }
            
                # Crea e allena la rete
                net = nn.NeuralNetwork(network_structure, **nn_params)

                # Test forward con un esempio
                if debug_mode:
                    print("  Test forward con un esempio...")
                    test_input = self.X_train[0]
                    print(f"    Test input shape: {test_input.shape}")
                    output = net.forward(test_input)
                    print(f"    Output shape: {output.shape}")

                # Parametri per fit
                fit_params = {'epochs': params.get('epochs', 1000),
                'batch_size': params.get('batch_size', 4),
                'patience': params.get('patience', 50),
                'verbose': False
                }
            
                net.fit(self.X_train, self.y_train, self.X_val, self.y_val, **fit_params)
            
                # Calcola accuratezza su validation set
                y_pred = net.predict(self.X_val)
            
                if output_size == 1:  # Classificazione binaria
                    y_pred_class = np.where(y_pred >= 0.5, 1, 0)
                    accuracy = np.mean(y_pred_class == self.y_val) * 100
                else:  # Regressione
                    # Per regressione usiamo negative MEE (più alto è meglio)
                    accuracy = -np.sqrt(np.mean((y_pred - self.y_val) ** 2))
            
                    accuracies.append(accuracy)
            
                # Aggiorna i migliori parametri
                    current_acc = np.mean(accuracies)
                    if current_acc > self.best_score:
                        self.best_score = current_acc
                        self.best_params = params.copy()
                        self.best_params['network_structure'] = network_structure
                        self.best_params['hidden_structure'] = network_structure[1:-1]  # Solo hidden layers
            except Exception as e:
                print(f"    Errore durante il trial {trial+1}: {e}")
                # In caso di errore, assegna una accuracy molto bassa
                accuracies.append(-999)
                continue
    
        if not accuracies:  # Se tutti i trials hanno fallito
            return 0, 0
    
        return np.mean(accuracies), np.std(accuracies)
        
    
    def enhanced_dichotomic_search(self, param_ranges, n_iterations=3, trials_per_config=3):
        """Ricerca dicotomica migliorata con più esplorazione"""
        print("=" * 60)
        print("RICERCA DICOTOMICA MIGLIORATA")
        print("=" * 60)
    
        # Parametri fissi per la ricerca
        fixed_params = {'algorithm': 'sgd', 'activation_type': 'sigmoid',
        'loss_type': 'half_mse',
        'epochs': 1000,'patience': 50
        }
    
        for iteration in range(n_iterations):
            print(f"\n{'='*40}")
            print(f"Iterazione {iteration + 1}/{n_iterations}")
            print(f"{'='*40}")
        
            # Per ogni parametro, testa 5 punti invece di 3
            test_points = {}
            for param, (low, high) in param_ranges.items():
                if param in ['hidden_layers', 'batch_size']:
                    # Per parametri interi
                    points = [low, low + (high-low)//4, low + (high-low)//2, 
                         low + 3*(high-low)//4, high]
                    test_points[param] = [int(p) for p in points]
                elif param == 'eta':
                    # Scala logaritmica per eta
                    low_log, high_log = np.log10(low), np.log10(high)
                    points_log = np.linspace(low_log, high_log, 5)
                    test_points[param] = [10**p for p in points_log]
                else:
                    # Per parametri float
                    points = np.linspace(low, high, 5)
                    test_points[param] = points.tolist()
        
            # Testa tutte le combinazioni
            best_accuracy = -np.inf
            best_point = None
        
            param_names = list(param_ranges.keys())
            param_values = [test_points[p] for p in param_names]
        
            total_combinations = np.prod([len(v) for v in param_values])
            print(f"Testando {total_combinations} combinazioni...")
        
            for i, combo in enumerate(itertools.product(*param_values)):
                params = dict(zip(param_names, combo))
                params.update(fixed_params)
            
                # Aggiungi struttura hidden layer complessa
                hidden_layers = params.get('hidden_layers', 1)
                hidden_neurons = params.get('hidden_neurons', 4)
            
                # Prova diverse strutture per lo stesso numero di neuroni totali
                total_neurons = hidden_layers * hidden_neurons
                structures_to_try = []
            
                # Genera diverse distribuzioni di neuroni
                if hidden_layers == 1:
                    structures_to_try = [[hidden_neurons]]
                elif hidden_layers == 2:
                    structures_to_try = [
                    [hidden_neurons, hidden_neurons],
                    [int(hidden_neurons*2), int(hidden_neurons//2)],
                    [int(hidden_neurons//2), int(hidden_neurons*2)]
                ]
                else:  # 3+ layers
                    structures_to_try = [
                    [hidden_neurons] * hidden_layers,
                    [int(hidden_neurons*2), hidden_neurons, int(hidden_neurons//2)],
                    [int(hidden_neurons//2), hidden_neurons, int(hidden_neurons*2)]
                ]
            
                for hidden_structure in structures_to_try:
                    # Converti tutti i valori a interi
                    hidden_structure = [int(n) for n in hidden_structure]
                
                    params_with_structure = params.copy()
                    params_with_structure['hidden_structure'] = hidden_structure
                
                try:
                    accuracy, std = self._evaluate_params(
                        params_with_structure, 
                        trials=trials_per_config
                    )
                    
                    print(f"  Config {i+1}/{total_combinations}: "
                          f"struttura={hidden_structure}, "
                          f"eta={params['eta']:.4f}, "
                          f"batch={params['batch_size']}, "
                          f"acc={accuracy:.2f}% ± {std:.2f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_point = params_with_structure.copy()
                        
                        self.search_history.append({
                            'params': params_with_structure.copy(),
                            'accuracy': accuracy,
                            'std': std,
                            'iteration': iteration
                        })
                        
                except Exception as e:
                    print(f"  Config {i+1}/{total_combinations}: ERRORE - {e}")
                    continue
        
        # AGGIUNGO QUESTO CONTROLLO: Se best_point è ancora None, crea un punto di default
        if best_point is None:
            print("Attenzione: nessuna configurazione valida trovata. Uso parametri di default.")
            best_point = {
                'eta': 0.1,
                'batch_size': 8,
                'hidden_layers': 2,
                'hidden_neurons': 8,
                'hidden_structure': [8, 4],
                'algorithm': 'sgd',
                'activation_type': 'sigmoid',
                'loss_type': 'half_mse',
                'epochs': 1000,
                'patience': 50
            }
            best_accuracy = 0  # O qualche valore di default
        
        # Restringe i range in modo più intelligente
        for param in param_ranges.keys():
            # Usa il valore da best_point o un valore di default
            if param in best_point:
                current_val = best_point[param]
            else:
                # Calcola il punto medio del range
                current_val = (param_ranges[param][0] + param_ranges[param][1]) / 2
                
            current_range = param_ranges[param]
            
            # Riduci il range del 60% invece del 50%
            range_size = current_range[1] - current_range[0]
            new_low = max(current_range[0], current_val - range_size * 0.3)
            new_high = min(current_range[1], current_val + range_size * 0.3)
            
            if param in ['hidden_layers', 'batch_size']:
                param_ranges[param] = [int(new_low), int(new_high)]
            else:
                param_ranges[param] = [new_low, new_high]
        
        print(f"\nMigliori parametri dopo iterazione {iteration + 1}:")
        print(f"  Hidden structure: {best_point.get('hidden_structure', [])}")
        print(f"  Eta: {best_point.get('eta', 0.1):.6f}")
        print(f"  Batch size: {best_point.get('batch_size', 8)}")
        print(f"  Accuracy: {best_accuracy:.2f}%")
    
    def aggressive_refinement(self, base_params, n_trials=20):
        """Ricerca aggressiva intorno ai migliori parametri"""
        print("\n" + "=" * 60)
        print("RAFFINAMENTO AGGRESSIVO")
        print("=" * 60)
        
        best_accuracy = -np.inf
        best_params = base_params.copy()
        
        # Prova diverse combinazioni aggressive
        algorithms = ['sgd', 'rprop', 'quickprop']
        activations = ['sigmoid', 'tanh']
        losses = ['half_mse', 'mae', 'huber', 'log_cosh']
        
        # Genera combinazioni random
        for trial in range(n_trials):
            params = base_params.copy()
            
            # Modifica i parametri in modo random
            params['algorithm'] = np.random.choice(algorithms)
            params['activation_type'] = np.random.choice(activations)
            params['loss_type'] = np.random.choice(losses)
            
            # Modifica eta leggermente
            if 'eta' in params:
                params['eta'] = params['eta'] * np.random.uniform(0.8, 1.2)
                params['eta'] = max(0.001, min(0.5, params['eta']))
            
            # Prova diverse strutture
            current_structure = params.get('hidden_structure', [params.get('hidden_neurons', 4)])
            new_structure = []
            for neurons in current_structure:
                # Modifica del ±25% e converti a int
                new_neurons = int(neurons * np.random.uniform(0.75, 1.25))
                new_neurons = max(2, min(32, new_neurons))
                new_structure.append(int(new_neurons))  # Converti a int
            
            params['hidden_structure'] = new_structure
            
            # Aggiungi/rimuovi layer con probabilità 0.3
            if np.random.random() < 0.3 and len(new_structure) < 5:
                new_structure.append(np.random.randint(2, 16))
                params['hidden_structure'] = new_structure
            elif np.random.random() < 0.3 and len(new_structure) > 1:
                params['hidden_structure'] = new_structure[:-1]
            
            # Valuta
            accuracy, std = self._evaluate_params(params, trials=3)
            
            print(f"  Trial {trial+1}: "
                  f"algo={params['algorithm']}, "
                  f"act={params['activation_type']}, "
                  f"loss={params['loss_type']}, "
                  f"struttura={params['hidden_structure']}, "
                  f"acc={accuracy:.2f}% ± {std:.2f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params.copy()
                
                self.search_history.append({
                    'params': params.copy(),
                    'accuracy': accuracy,
                    'std': std,
                    'type': 'aggressive_refinement'
                })
        
        # Aggiorna i migliori parametri globali
        if best_accuracy > self.best_score:
            self.best_score = best_accuracy
            self.best_params = best_params
        
        return best_params
    
    def find_optimal_structure(self, n_explorations=10):
        """Ricerca specifica per la struttura ottimale"""
        print("\n" + "=" * 60)
        print("RICERCA STRUTTURA OTTIMALE")
        print("=" * 60)
        
        base_params = self.best_params.copy() if self.best_params else {
            'eta': 0.1,
            'batch_size': 8,
            'algorithm': 'sgd',
            'activation_type': 'sigmoid',
            'loss_type': 'half_mse',
            'epochs': 1500,
            'patience': 50
        }
        
        # Strutture da testare (basate sulla letteratura per Monk problems)
        structures_to_test = [
            [4],                    # 1 layer, 4 neuroni
            [8],                    # 1 layer, 8 neuroni
            [4, 4],                 # 2 layer, 4 neuroni ciascuno
            [8, 4],                 # 2 layer, 8 e 4 neuroni
            [4, 4, 4],              # 3 layer, 4 neuroni ciascuno
            [8, 8, 4],              # 3 layer, 8, 8, 4 neuroni
            [16, 8],                # 2 layer, 16 e 8 neuroni
            [12, 6, 3],             # 3 layer, decrescente
            [6, 12, 6],             # 3 layer, a forma di clessidra
        ]
        
        best_accuracy = -np.inf
        best_structure = None
        
        for i, structure in enumerate(structures_to_test):
            params = base_params.copy()
            params['hidden_structure'] = structure
            
            # Adatta eta in base alla complessità
            complexity = sum(structure) / len(structure)
            params['eta'] = max(0.01, min(0.2, 0.1 / np.sqrt(complexity)))
            
            accuracy, std = self._evaluate_params(params, trials=3)
            
            print(f"  Struttura {i+1}/{len(structures_to_test)}: {structure}")
            print(f"    Accuracy: {accuracy:.2f}% ± {std:.2f}")
            print(f"    Eta usato: {params['eta']:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_structure = structure
                base_params = params.copy()  # Usa i migliori parametri come base
        
        # Aggiorna i migliori parametri
        if best_accuracy > self.best_score:
            self.best_score = best_accuracy
            self.best_params = base_params
            self.best_params['hidden_structure'] = best_structure
        
        return best_structure
    
    def final_evaluation_with_cross_val(self, n_trials=300, n_folds=3):
        """Valutazione finale con cross-validazione approssimata"""
        print("\n" + "=" * 60)
        print("VALUTAZIONE FINALE CON MULTIPLE TRIAL")
        print("=" * 60)
        
        if not self.best_params:
            print("Nessun parametro trovato. Usa default.")
            self.best_params = {
                'eta': 0.1,
                'batch_size': 8,
                'algorithm': 'rprop',
                'activation_type': 'tanh',
                'loss_type': 'half_mse',
                'hidden_structure': [8, 4],
                'epochs': 2000,
                'patience': 100,
                'l2_lambda': 0.0001,
                'momentum': 0.9
            }
        
        print(f"\nParametri ottimali trovati:")
        for key, value in self.best_params.items():
            if key not in ['network_structure', 'hidden_structure']:
                print(f"  {key}: {value}")
        
        if 'hidden_structure' in self.best_params:
            print(f"  Struttura hidden layers: {self.best_params['hidden_structure']}")
            print(f"  Numero di hidden layers: {len(self.best_params['hidden_structure'])}")
            print(f"  Neuroni per layer: {self.best_params['hidden_structure']}")
        
        # DEBUG: Stampa la forma dei dati
        print(f"\nDEBUG - Forma dei dati:")
        print(f"  X_train shape: {self.X_train.shape}")
        print(f"  X_val shape: {self.X_val.shape}")
        print(f"  X_test shape: {self.X_test.shape}")
        accuracies_val = []
        accuracies_test = []
        
        print(f"\nEseguendo {n_trials} trials...")
        start_time = time.time()
        
        for trial in range(n_trials):
            if (trial + 1) % 50 == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / (trial + 1) * (n_trials - trial - 1)
                print(f"  Trial {trial + 1}/{n_trials} - "
                      f"Tempo rimanente: {remaining/60:.1f} min")
            
            # Costruisci struttura rete
            input_size = self.X_train.shape[1]
            output_size = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1
            
            network_structure = [input_size]
            if 'hidden_structure' in self.best_params:
                network_structure.extend(self.best_params['hidden_structure'])
            else:
                hidden_layers = self.best_params.get('hidden_layers', 2)
                hidden_neurons = self.best_params.get('hidden_neurons', 8)
                for _ in range(hidden_layers):
                    network_structure.append(hidden_neurons)
            network_structure.append(output_size)
            print(f"  Trial {trial+1}: Struttura rete: {network_structure}")
        
            # DEBUG: Verifica che la struttura sia valida
            if network_structure[0] != input_size:
                print(f"  ERRORE: input_size mismatch! Rete attende {network_structure[0]}, dati hanno {input_size}")
            continue

            # Crea rete
            net = nn.NeuralNetwork(
                network_structure,
                eta=self.best_params.get('eta', 0.1),
                loss_type=self.best_params.get('loss_type', 'half_mse'),
                algorithm=self.best_params.get('algorithm', 'sgd'),
                activation_type=self.best_params.get('activation_type', 'sigmoid'),
                l2_lambda=self.best_params.get('l2_lambda', 0.0),
                momentum=self.best_params.get('momentum', 0.0),
                weight_initializer=self.best_params.get('weight_initializer', 'def')
            )
            
            # Allena
            net.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=self.best_params.get('epochs', 2000),
                batch_size=self.best_params.get('batch_size', 8),
                patience=self.best_params.get('patience', 100),
                verbose=False
            )
            
            # Valuta
            try:
                y_pred_val = net.predict(self.X_val)
                y_pred_test = net.predict(self.X_test)
            
                if output_size == 1:  # Classificazione binaria
                    y_pred_val_class = np.where(y_pred_val >= 0.5, 1, 0)
                    y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
                
                    val_accuracy = np.mean(y_pred_val_class == self.y_val) * 100
                    test_accuracy = np.mean(y_pred_test_class == self.y_test) * 100
                else:  # Regressione
                    val_accuracy = -np.sqrt(np.mean((y_pred_val - self.y_val) ** 2))
                    test_accuracy = -np.sqrt(np.mean((y_pred_test - self.y_test) ** 2))
            
                accuracies_val.append(val_accuracy)
                accuracies_test.append(test_accuracy)
                print(f"  Trial {trial+1}: Val={val_accuracy:.2f}%, Test={test_accuracy:.2f}%")
            
            except Exception as e:
                print(f"  ERRORE nel trial {trial+1}: {e}")
            # Salta questo trial ma continua
            continue
        
        # Calcola statistiche
        mean_val = np.mean(accuracies_val)
        std_val = np.std(accuracies_val)
        mean_test = np.mean(accuracies_test)
        std_test = np.std(accuracies_test)
        
        print("\n" + "=" * 60)
        print("RISULTATI FINALI")
        print("=" * 60)
        
        print(f"\nParametri ottimali:")
        for key, value in self.best_params.items():
            if key not in ['network_structure']:
                print(f"  {key}: {value}")
        
        print(f"\nStruttura della rete:")
        print(f"  Input: {input_size} neuroni")
        if 'hidden_structure' in self.best_params:
            for i, neurons in enumerate(self.best_params['hidden_structure']):
                print(f"  Hidden layer {i+1}: {neurons} neuroni")
        print(f"  Output: {output_size} neuroni")
        
        print(f"\nPerformance su Validation Set ({n_trials} trials):")
        print(f"  Accuracy media: {mean_val:.2f}%")
        print(f"  Deviazione standard: {std_val:.2f}%")
        print(f"  Min: {np.min(accuracies_val):.2f}%")
        print(f"  Max: {np.max(accuracies_val):.2f}%")
        
        print(f"\nPerformance su Test Set ({n_trials} trials):")
        print(f"  Accuracy media: {mean_test:.2f}%")
        print(f"  Deviazione standard: {std_test:.2f}%")
        print(f"  Min: {np.min(accuracies_test):.2f}%")
        print(f"  Max: {np.max(accuracies_test):.2f}%")
        
        # Calcola intervallo di confidenza 95%
        ci_val = 1.96 * std_val / np.sqrt(n_trials)
        ci_test = 1.96 * std_test / np.sqrt(n_trials)
        
        print(f"\nIntervalli di confidenza 95%:")
        print(f"  Validation: ({mean_val-ci_val:.2f}%, {mean_val+ci_val:.2f}%)")
        print(f"  Test: ({mean_test-ci_test:.2f}%, {mean_test+ci_test:.2f}%)")
        
        return {
            'best_params': self.best_params,
            'val_mean': mean_val,
            'val_std': std_val,
            'test_mean': mean_test,
            'test_std': std_test,
            'val_accuracies': accuracies_val,
            'test_accuracies': accuracies_test
        }


def run_advanced_monk_search(monk_dataset=3):
    """Ricerca avanzata per dataset Monk"""
    # Carica dati
    if monk_dataset == 1:
        X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk1(
            one_hot=True, dataset_shuffle=True
        )
    elif monk_dataset == 2:
        X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk2(
            one_hot=True, dataset_shuffle=True
        )
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk3(
            one_hot=True, dataset_shuffle=True
        )
    
    print(f"\n{'='*60}")
    print(f"DATASET MONK-{monk_dataset}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}")
    
    # Normalizzazione
    X_train_norm, X_val_norm, X_test_norm = data.normalize_dataset(
        X_train, X_val, X_test, 0, 1
    )
    
    # Inizializza grid search
    search = GridSearch(
        X_train_norm, y_train, 
        X_val_norm, y_val, 
        X_test_norm, y_test
    )
    
    # FASE 1: Ricerca dicotomica migliorata
    print("\n" + "="*60)
    print("FASE 1: RICERCA DICOTOMICA MIGLIORATA")
    print("="*60)
    
    param_ranges = {
        'eta': [0.001, 0.3],           # Range più ampio
        'batch_size': [4, 32],          # Batch size più ragionevole
        'hidden_layers': [1, 3],        # Fino a 3 hidden layers
        'hidden_neurons': [2, 16]       # Neuroni per layer
    }
    
    search.enhanced_dichotomic_search(
        param_ranges, 
        n_iterations=2, 
        trials_per_config=2
    )
    
    # FASE 2: Ricerca struttura ottimale
    optimal_structure = search.find_optimal_structure(n_explorations=15)
    
    # FASE 3: Raffinamento aggressivo
    best_params = search.aggressive_refinement(
        search.best_params, 
        n_trials=25
    )
    
    # FASE 4: Valutazione finale
    results = search.final_evaluation_with_cross_val(
        n_trials=300, 
        n_folds=3
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Grid Search per Neural Network')
    parser.add_argument('--dataset', type=str, default='monk3',
                       choices=['monk1', 'monk2', 'monk3'],
                       help='Dataset Monk da usare')
    
    args = parser.parse_args()
    monk_num = int(args.dataset[-1])
    
    results = run_advanced_monk_search(monk_num)
    
    # Salva risultati
    import json
    with open(f'advanced_grid_search_monk{monk_num}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRisultati salvati in advanced_grid_search_monk{monk_num}.json")
    
    # Suggerimenti per miglioramenti ulteriori
    print("\n" + "="*60)
    print("SUGGERIMENTI PER MIGLIORARE:")
    print("="*60)
    print("1. Se l'accuracy è ancora bassa, prova ad aumentare gli epochs a 3000+")
    print("2. Prova l'algoritmo 'rprop' con eta_plus=1.2, eta_minus=0.5")
    print("3. Per Monk3 (con noise), aggiungi regolarizzazione L2 (0.001-0.01)")
    print("4. Prova funzioni di attivazione 'tanh' con inizializzazione 'xavier'")
    print("5. Considera early stopping più aggressivo (patience=20)")