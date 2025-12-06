import numpy as np
import time

# Import dei tuoi file
from data_manipulation import return_monk1
from grid_search_cv import GridSearch

def main():
    num_trials = 500
    print("="*70)
    print(f"GRID SEARCH + {num_trials} TRIAL SU MONK1")
    print("="*70)
    
    print("\n[1] Caricamento dataset MONK1...")
    X_train, y_train, X_test, y_test = return_monk1(dataset_shuffle=False, one_hot=True)
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    print("\n[2] Esecuzione Grid Search...")
    print("-"*70)
    
    gs = GridSearch(cv_folds=5, verbose=True, results_dir='grid_search_results')
    
    gs.lr_range = [0.01, 0.25]
    gs.hidden_range = [2, 6]
    gs.epochs_range = [300, 600]
    
    # Esegui ricerca
    best_params, cv_acc, cv_std = gs._dichotomic_search(
        X_train, y_train,
        max_iteration=4,
        n_points=5
    )
    
    print(f"\n[3] {num_trials} Trial con parametri ottimali...")
    print("-"*70)
    
    import neural_network as nn
    
    accuracies = []
    
    print(f"Parametri usati per {num_trials} trial:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nInizio {num_trials} addestramenti...")
    start_time = time.time()
    
    for i in range(num_trials):
        if (i + 1) % 5 == 0:
            print(f"  Trial {i+1}/{num_trials}...")
        
        # Crea rete con struttura dai parametri ottimali
        input_size = X_train.shape[1]
        structure = [input_size, best_params['hidden_neurons'], 1]
        net = nn.NeuralNetwork(structure, eta=best_params['learning_rate'])
        
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
                epochs=best_params['epochs'], verbose=False)
        
        # Valuta sul test set
        predictions = net.predict(X_test)
        pred_classes = (predictions >= 0.5).astype(int)
        accuracy = np.mean(pred_classes == y_test) * 100
        accuracies.append(accuracy)
    
    elapsed_time = time.time() - start_time
    
    # 4. Calcola statistiche
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print(f"\n[4] Statistiche su {num_trials} trial:")
    print("-"*70)
    print(f"Tempo totale: {elapsed_time:.1f} secondi")
    print(f"\nAccuracy media: {mean_acc:.2f}%")
    print(f"Deviazione standard: {std_acc:.2f}%")
    print(f"Minimo: {min_acc:.2f}%")
    print(f"Massimo: {max_acc:.2f}%")
    
    print("\n" + "="*70)
    print("RISULTATO FINALE")
    print("="*70)
    print(f"Parametri ottimali: lr={best_params['learning_rate']:.3f}, "
          f"hidden={best_params['hidden_neurons']}, epochs={best_params['epochs']}")
    print(f"\nAccuracy su {num_trials} trials: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Range: {mean_acc - std_acc:.2f}% - {mean_acc + std_acc:.2f}%")
    

if __name__ == "__main__":
    main()