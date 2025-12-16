import numpy as np
import itertools
import random
import json
import time
import neural_network as nn
import data_manipulation as data

class GridSearch:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, dataset_type='monk'):
        """
        Inizializza il grid search
        
        Parameters:
        -----------
        dataset_type: 'monk' per classificazione binaria, 'cup' per regressione
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_type = dataset_type
        self.best_params = None
        self.best_score = -np.inf if dataset_type == 'monk' else np.inf
        self.best_accuracy_val = -np.inf
        self.best_accuracy_train = -np.inf
        self.history = []
    
    def evaluate_configuration(self, params, n_trials=1):
        # Valuta una configurazione di parametri su n_trials
        accuracies_val = []
        accuracies_train = []
    
        for trial in range(n_trials):
            try:
                # Determina la struttura della rete
                input_size = self.X_train.shape[1]
                output_size = 1 if self.dataset_type == 'monk' else self.y_train.shape[1]
            
                # Costruisci la struttura della rete - AUMENTA LA DIMENSIONE!
                if 'hidden_structure' in params:
                    network_structure = [input_size] + params['hidden_structure'] + [output_size]
                else:
                    # Struttura più grande di default
                    network_structure = [input_size, 8, 4, output_size]  # 2 hidden layers
            
                # Crea la rete neurale - RIDUCI IL LEARNING RATE!
                net = nn.NeuralNetwork(
                    network_structure=network_structure,
                    eta=params.get('eta', 0.01),  # 0.01 invece di 0.1
                    loss_type=params.get('loss_type', 'half_mse'),
                    algorithm=params.get('algorithm', 'sgd'),
                    activation_type=params.get('activation_type', 'sigmoid'),
                    l2_lambda=params.get('l2_lambda', 0.0),
                    momentum=params.get('momentum', 0.0),
                    weight_initializer=params.get('weight_initializer', 'def')
                )
            
                # Allena la rete - USA MENO EPOCHE PER TEST
                net.fit(
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    epochs=params.get('epochs', 200),  # 200 invece di 1000
                    batch_size=params.get('batch_size', 8),
                    patience=params.get('patience', 30),  # 30 invece di 50
                    verbose=False
                )
            
                # Valuta su SUBSET del training set
                subset_size = min(100, len(self.X_train))  # Massimo 100 esempi
                y_pred_train = net.predict(self.X_train[:subset_size])
                if self.dataset_type == 'monk':
                    y_pred_train_class = np.where(y_pred_train >= 0.5, 1, 0)
                    accuracy_train = np.mean(y_pred_train_class == self.y_train[:subset_size]) * 100
                else:
                    accuracy_train = -np.sqrt(np.mean((y_pred_train - self.y_train[:subset_size]) ** 2))
            
                # Valuta su validation set
                y_pred_val = net.predict(self.X_val)
                if self.dataset_type == 'monk':
                    y_pred_val_class = np.where(y_pred_val >= 0.5, 1, 0)
                    accuracy_val = np.mean(y_pred_val_class == self.y_val) * 100
                else:
                    accuracy_val = -np.sqrt(np.mean((y_pred_val - self.y_val) ** 2))
            
                accuracies_train.append(accuracy_train)
                accuracies_val.append(accuracy_val)
            
            except Exception as e:
                print(f"    Errore nel trial {trial}: {e}")
                continue
    
        if not accuracies_val:
            return 0, 0, 0, 0
    
        return (np.mean(accuracies_val), np.std(accuracies_val),
                np.mean(accuracies_train), np.std(accuracies_train))
    
    def dichotomic_search(self, param_grid, n_iterations=3, trials_per_config=3):
        """
        Ricerca dicotomica sugli iperparametri numerici
        
        Parameters:
        -----------
        param_grid: dizionario con i range degli iperparametri
                    es: {'eta': [0.001, 0.1], 'batch_size': [4, 32], ...}
        """
        print("=" * 70)
        print("RICERCA DICOTOMICA SUGLI IPERPARAMETRI")
        print("=" * 70)
        
        current_ranges = param_grid.copy()
        
        for iteration in range(n_iterations):
            print(f"\nIterazione {iteration + 1}/{n_iterations}")
            print("-" * 40)
            
            # Per ogni parametro, genera 3 valori (min, medio, max)
            param_values = {}
            for param, (low, high) in current_ranges.items():
                if param in ['batch_size', 'epochs', 'num_hidden_layers', 'neurons_per_layer']:
                    # Parametri interi
                    mid = int((low + high) / 2)
                    param_values[param] = [low, mid, high]
                else:
                    # Parametri float (scala logaritmica per eta)
                    if param == 'eta':
                        low_log, high_log = np.log10(low), np.log10(high)
                        mid_log = (low_log + high_log) / 2
                        param_values[param] = [10**low_log, 10**mid_log, 10**high_log]
                    else:
                        mid = (low + high) / 2
                        param_values[param] = [low, mid, high]
            
            # Genera tutte le combinazioni
            param_names = list(param_values.keys())
            all_combinations = list(itertools.product(*[param_values[p] for p in param_names]))
            
            print(f"Testando {len(all_combinations)} combinazioni...")
            
            best_score_iter = -np.inf if self.dataset_type == 'monk' else np.inf
            best_params_iter = None
            
            for i, combo in enumerate(all_combinations):
                params = dict(zip(param_names, combo))
                
                # Aggiungi parametri fissi
                params.update({
                    'algorithm': 'sgd',
                    'activation_type': 'sigmoid',
                    'loss_type': 'half_mse',
                    'patience': 50
                })
                
                # Valuta la configurazione
                accuracy_val, std_val, accuracy_train, std_train = self.evaluate_configuration(
                    params, trials_per_config
                )
                
                score = accuracy_val  # Usiamo validation accuracy come score
                
                print(f"  Config {i+1}:")
                print(f"    Params: {params}")
                print(f"    Val Acc: {accuracy_val:.2f}% ± {std_val:.2f}")
                print(f"    Train Acc: {accuracy_train:.2f}% ± {std_train:.2f}")
                
                # Aggiorna il miglior score
                if self.dataset_type == 'monk':
                    if score > best_score_iter:
                        best_score_iter = score
                        best_params_iter = params.copy()
                else:
                    if score < best_score_iter:  # Per CUP, score più basso (MEE negativo più alto) è meglio
                        best_score_iter = score
                        best_params_iter = params.copy()
                
                # Salva nella history
                self.history.append({
                    'params': params.copy(),
                    'accuracy_val': accuracy_val,
                    'std_val': std_val,
                    'accuracy_train': accuracy_train,
                    'std_train': std_train,
                    'iteration': iteration
                })
            
            # Restringe i range per la prossima iterazione
            for param in current_ranges.keys():
                if param in best_params_iter:
                    current_val = best_params_iter[param]
                    current_range = current_ranges[param]
                    
                    # Calcola nuovo range (50% più stretto)
                    range_width = current_range[1] - current_range[0]
                    new_low = max(current_range[0], current_val - range_width * 0.25)
                    new_high = min(current_range[1], current_val + range_width * 0.25)
                    
                    current_ranges[param] = [new_low, new_high]
            
            print(f"\nMigliori parametri dopo iterazione {iteration + 1}:")
            for key, value in best_params_iter.items():
                print(f"  {key}: {value}")
            print(f"  Validation Accuracy: {best_score_iter:.2f}%")
        
        # Aggiorna i migliori parametri globali
        if self.dataset_type == 'monk':
            if best_score_iter > self.best_score:
                self.best_score = best_score_iter
                self.best_params = best_params_iter
                self.best_accuracy_val = best_score_iter
        else:
            if best_score_iter < self.best_score:
                self.best_score = best_score_iter
                self.best_params = best_params_iter
                self.best_accuracy_val = best_score_iter
    
    def refine_search(self, base_params, refinement_grid, trials_per_config=5):
        """
        Ricerca di raffinamento sugli iperparametri categorici e strutturali
        
        Parameters:
        -----------
        refinement_grid: dizionario con le opzioni da testare
                        es: {'algorithm': ['sgd', 'rprop', 'quickprop'],
                             'hidden_structure': [[4], [8], [4,4], [8,4]]}
        """
        print("\n" + "=" * 70)
        print("RAFFINAMENTO RICERCA IPERPARAMETRI")
        print("=" * 70)
        
        # Genera tutte le combinazioni dalla griglia di raffinamento
        param_names = list(refinement_grid.keys())
        param_values = [refinement_grid[p] for p in param_names]
        
        total_combinations = np.prod([len(v) for v in param_values])
        print(f"Testando {total_combinations} combinazioni...")
        
        best_score_refine = -np.inf if self.dataset_type == 'monk' else np.inf
        best_params_refine = None
        
        for i, combo in enumerate(itertools.product(*param_values)):
            # Crea i parametri combinando base_params con le nuove combinazioni
            params = base_params.copy()
            params.update(dict(zip(param_names, combo)))
            
            # Valuta la configurazione
            accuracy_val, std_val, accuracy_train, std_train = self.evaluate_configuration(
                params, trials_per_config
            )
            
            score = accuracy_val  # Usiamo validation accuracy come score
            
            print(f"\nConfigurazione {i+1}/{total_combinations}:")
            print(f"  Parametri: {combo}")
            print(f"  Validation Accuracy: {accuracy_val:.2f}% ± {std_val:.2f}")
            print(f"  Training Accuracy: {accuracy_train:.2f}% ± {std_train:.2f}")
            
            # Aggiorna il miglior score
            if self.dataset_type == 'monk':
                if score > best_score_refine:
                    best_score_refine = score
                    best_params_refine = params.copy()
                    self.best_accuracy_val = score
                    self.best_accuracy_train = accuracy_train
            else:
                if score < best_score_refine:
                    best_score_refine = score
                    best_params_refine = params.copy()
                    self.best_accuracy_val = score
                    self.best_accuracy_train = accuracy_train
            
            # Salva nella history
            self.history.append({
                'params': params.copy(),
                'accuracy_val': accuracy_val,
                'std_val': std_val,
                'accuracy_train': accuracy_train,
                'std_train': std_train,
                'type': 'refinement'
            })
        
        # Aggiorna i migliori parametri globali
        if best_params_refine:
            if self.dataset_type == 'monk':
                if best_score_refine > self.best_score:
                    self.best_score = best_score_refine
                    self.best_params = best_params_refine
            else:
                if best_score_refine < self.best_score:
                    self.best_score = best_score_refine
                    self.best_params = best_params_refine
        
        return best_params_refine
    
    def final_evaluation(self, n_trials=300):
        """
        Valutazione finale con i migliori parametri trovati
        
        Returns:
        --------
        dict con i risultati
        """
        print("\n" + "=" * 70)
        print("VALUTAZIONE FINALE CON MIGLIORI PARAMETRI")
        print("=" * 70)
        
        if not self.best_params:
            print("Nessun parametro ottimale trovato. Uso parametri di default.")
            self.best_params = {
                'eta': 0.1,
                'batch_size': 8,
                'epochs': 1000,
                'algorithm': 'sgd',
                'activation_type': 'sigmoid',
                'loss_type': 'half_mse',
                'hidden_structure': [8, 4],
                'patience': 50
            }
        
        # Stampa i migliori parametri
        print("\nMIGLIORI PARAMETRI TROVATI:")
        print("-" * 40)
        for key, value in self.best_params.items():
            print(f"{key:20}: {value}")
        
        # Valutazione su n_trials
        print(f"\nEseguendo {n_trials} trials con i migliori parametri...")
        
        accuracies_val = []
        accuracies_train = []
        accuracies_test = []
        
        for trial in range(n_trials):
            if (trial + 1) % 50 == 0:
                print(f"  Trial {trial + 1}/{n_trials}")
            
            try:
                # Determina la struttura della rete
                input_size = self.X_train.shape[1]
                output_size = 1 if self.dataset_type == 'monk' else self.y_train.shape[1]
                
                # Costruisci la struttura della rete
                if 'hidden_structure' in self.best_params:
                    network_structure = [input_size] + self.best_params['hidden_structure'] + [output_size]
                else:
                    # Costruisci struttura basata su numero di layer e neuroni
                    network_structure = [input_size]
                    if 'num_hidden_layers' in self.best_params and 'neurons_per_layer' in self.best_params:
                        for _ in range(self.best_params['num_hidden_layers']):
                            network_structure.append(self.best_params['neurons_per_layer'])
                    network_structure.append(output_size)
                
                # Crea la rete neurale
                net = nn.NeuralNetwork(
                    network_structure=network_structure,
                    eta=self.best_params.get('eta', 0.1),
                    loss_type=self.best_params.get('loss_type', 'half_mse'),
                    algorithm=self.best_params.get('algorithm', 'sgd'),
                    activation_type=self.best_params.get('activation_type', 'sigmoid'),
                    l2_lambda=self.best_params.get('l2_lambda', 0.0),
                    momentum=self.best_params.get('momentum', 0.0),
                    weight_initializer=self.best_params.get('weight_initializer', 'def')
                )
                
                # Allena la rete
                net.fit(
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    epochs=self.best_params.get('epochs', 1000),
                    batch_size=self.best_params.get('batch_size', 4),
                    patience=self.best_params.get('patience', 50),
                    verbose=False
                )
                
                # Valuta su training set
                y_pred_train = net.predict(self.X_train)
                if self.dataset_type == 'monk':
                    y_pred_train_class = np.where(y_pred_train >= 0.5, 1, 0)
                    accuracy_train = np.mean(y_pred_train_class == self.y_train) * 100
                else:
                    accuracy_train = -np.sqrt(np.mean((y_pred_train - self.y_train) ** 2))
                
                # Valuta su validation set
                y_pred_val = net.predict(self.X_val)
                if self.dataset_type == 'monk':
                    y_pred_val_class = np.where(y_pred_val >= 0.5, 1, 0)
                    accuracy_val = np.mean(y_pred_val_class == self.y_val) * 100
                else:
                    accuracy_val = -np.sqrt(np.mean((y_pred_val - self.y_val) ** 2))
                
                # Valuta su test set
                y_pred_test = net.predict(self.X_test)
                if self.dataset_type == 'monk':
                    y_pred_test_class = np.where(y_pred_test >= 0.5, 1, 0)
                    accuracy_test = np.mean(y_pred_test_class == self.y_test) * 100
                else:
                    accuracy_test = -np.sqrt(np.mean((y_pred_test - self.y_test) ** 2))
                
                accuracies_train.append(accuracy_train)
                accuracies_val.append(accuracy_val)
                accuracies_test.append(accuracy_test)
                
            except Exception as e:
                print(f"  Errore nel trial {trial + 1}: {e}")
                continue
        
        # Calcola statistiche
        mean_train = np.mean(accuracies_train)
        std_train = np.std(accuracies_train)
        mean_val = np.mean(accuracies_val)
        std_val = np.std(accuracies_val)
        mean_test = np.mean(accuracies_test)
        std_test = np.std(accuracies_test)
        
        print("\n" + "=" * 70)
        print("RISULTATI FINALI")
        print("=" * 70)
        
        print(f"\nStruttura rete finale: {network_structure}")
        
        if self.dataset_type == 'monk':
            print(f"\nPerformance su {n_trials} trials:")
            print(f"  Training Set:    {mean_train:.2f}% ± {std_train:.2f}%")
            print(f"  Validation Set:  {mean_val:.2f}% ± {std_val:.2f}%")
            print(f"  Test Set:        {mean_test:.2f}% ± {std_test:.2f}%")
        else:
            print(f"\nPerformance su {n_trials} trials (MEE negativo, più alto è meglio):")
            print(f"  Training Set:    {mean_train:.4f} ± {std_train:.4f}")
            print(f"  Validation Set:  {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Test Set:        {mean_test:.4f} ± {std_test:.4f}")
        
        return {
            'best_params': self.best_params,
            'network_structure': network_structure,
            'train_mean': mean_train,
            'train_std': std_train,
            'val_mean': mean_val,
            'val_std': std_val,
            'test_mean': mean_test,
            'test_std': std_test
        }


def run_grid_search(dataset_name='monk3', one_hot=True, dataset_shuffle=True,
                   cup_train_size=250, cup_val_size=125, cup_test_size=125):
    """
    Esegue il grid search completo su un dataset specificato
    
    Parameters:
    -----------
    dataset_name: 'monk1', 'monk2', 'monk3', o 'cup'
    """
    print("=" * 70)
    print(f"GRID SEARCH PER NEURAL NETWORK - DATASET: {dataset_name.upper()}")
    print("=" * 70)
    
    # Carica il dataset appropriato
    if dataset_name.startswith('monk'):
        if '1' in dataset_name:
            X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk1(
                one_hot=one_hot, dataset_shuffle=dataset_shuffle
            )
            dataset_type = 'monk'
        elif '2' in dataset_name:
            X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk2(
                one_hot=one_hot, dataset_shuffle=dataset_shuffle
            )
            dataset_type = 'monk'
        else:  # monk3
            X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk3(
                one_hot=one_hot, dataset_shuffle=dataset_shuffle
            )
            dataset_type = 'monk'
        
        # Converti bool a float se necessario
        if X_train.dtype == bool:
            X_train = X_train.astype(float)
            X_val = X_val.astype(float)
            X_test = X_test.astype(float)
        
        print(f"Dati {dataset_name} caricati:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
    elif dataset_name == 'cup':
        X_train, y_train, X_val, y_val, X_test, y_test = data.return_CUP(
            dataset_shuffle=dataset_shuffle,
            train_size=cup_train_size,
            validation_size=cup_val_size,
            test_size=cup_test_size
        )
        dataset_type = 'cup'
        
        print(f"Dati CUP caricati:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")
    
    # Normalizza i dati
    X_train_norm, X_val_norm, X_test_norm = data.normalize_dataset(
        X_train, X_val, X_test, 0, 1
    )
    
    # Inizializza il grid search
    search = GridSearch(
        X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test,
        dataset_type=dataset_type
    )
    
    # FASE 1: Ricerca dicotomica sugli iperparametri numerici
    print("\n" + "=" * 70)
    print("FASE 1: RICERCA DICOTOMICA")
    print("=" * 70)
    
    if dataset_type == 'monk':
        param_grid = {
            'eta': [0.001, 0.2],
            'batch_size': [4, 32],
            'epochs': [300, 1500],
            'num_hidden_layers': [1, 3],
            'neurons_per_layer': [2, 16]
        }
    else:  # CUP
        param_grid = {
            'eta': [0.0001, 0.05],
            'batch_size': [8, 64],
            'epochs': [500, 2000],
            'num_hidden_layers': [1, 3],
            'neurons_per_layer': [4, 32]
        }
    
    search.dichotomic_search(param_grid, n_iterations=2, trials_per_config=2)
    
    # FASE 2: Ricerca di raffinamento
    print("\n" + "=" * 70)
    print("FASE 2: RAFFINAMENTO IPERPARAMETRI")
    print("=" * 70)
    
    # Definisci la griglia di raffinamento
    refinement_grid = {
        'algorithm': ['sgd', 'rprop', 'quickprop'],
        'activation_type': ['sigmoid', 'tanh'],
        'loss_type': ['half_mse', 'mae', 'huber', 'log_cosh'],
    }
    
    # Aggiungi diverse strutture di rete da testare
    input_size = X_train.shape[1]
    if dataset_type == 'monk':
        # Strutture per Monk (input size è 17 con one-hot)
        structures = [
            [4],                    # 1 hidden layer, 4 neuroni
            [8],                    # 1 hidden layer, 8 neuroni
            [4, 4],                 # 2 hidden layers, 4 neuroni ciascuno
            [8, 4],                 # 2 hidden layers, 8 e 4 neuroni
            [8, 8, 4],              # 3 hidden layers, 8, 8, 4 neuroni
            [12, 6],                # 2 hidden layers, 12 e 6 neuroni
            [16, 8, 4],             # 3 hidden layers, 16, 8, 4 neuroni
        ]
    else:  # CUP
        # Strutture per CUP (input size è 10)
        structures = [
            [8],                    # 1 hidden layer, 8 neuroni
            [16],                   # 1 hidden layer, 16 neuroni
            [8, 4],                 # 2 hidden layers, 8 e 4 neuroni
            [16, 8],                # 2 hidden layers, 16 e 8 neuroni
            [16, 8, 4],             # 3 hidden layers, 16, 8, 4 neuroni
            [24, 12, 6],            # 3 hidden layers, 24, 12, 6 neuroni
        ]
    
    refinement_grid['hidden_structure'] = structures
    
    # Aggiungi regolarizzazione L2 e momentum
    refinement_grid['l2_lambda'] = [0.0, 0.0001, 0.001]
    refinement_grid['momentum'] = [0.0, 0.5, 0.9]
    
    search.refine_search(search.best_params, refinement_grid, trials_per_config=3)
    
    # FASE 3: Valutazione finale
    print("\n" + "=" * 70)
    print("FASE 3: VALUTAZIONE FINALE")
    print("=" * 70)
    
    results = search.final_evaluation(n_trials=300)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid Search per Neural Network')
    parser.add_argument('--dataset', type=str, default='monk3',
                       choices=['monk1', 'monk2', 'monk3', 'cup'],
                       help='Dataset da usare')
    parser.add_argument('--no_one_hot', action='store_true',
                       help='Disabilita one-hot encoding (solo per Monk)')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='Disabilita shuffle dei dati')
    parser.add_argument('--train_size', type=int, default=250,
                       help='Training size per CUP')
    parser.add_argument('--val_size', type=int, default=125,
                       help='Validation size per CUP')
    parser.add_argument('--test_size', type=int, default=125,
                       help='Test size per CUP')
    
    args = parser.parse_args()
    
    results = run_grid_search(
        dataset_name=args.dataset,
        one_hot=not args.no_one_hot,
        dataset_shuffle=not args.no_shuffle,
        cup_train_size=args.train_size,
        cup_val_size=args.val_size,
        cup_test_size=args.test_size
    )
    
    # Salva i risultati
    with open(f'grid_search_results_{args.dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRisultati salvati in grid_search_results_{args.dataset}.json")