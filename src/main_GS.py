import numpy as np
import data_manipulation as data
from grid_search_cv import SimpleGridSearchCV
import time
import json

def train_final_model_multiple_runs(X_train, y_train, X_test, y_test, params, n_runs=10):
    import neural_network as nn
    
    all_accuracies = []
    input_size = X_train.shape[1]
    
    print(f"\nAllenamento finale ({n_runs} run indipendenti):")
    
    for run in range(n_runs):
        # Crea una nuova rete per ogni run
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
        
        # Allena
        net.fit(X_train, X_test, y_train, y_test, 
                epochs=params['epochs'])
        
        # Valuta sul test set
        predictions = net.predict(X_test)
        pred_classes = (predictions >= 0.5).astype(int)
        accuracy = np.mean(pred_classes == y_test) * 100
        
        all_accuracies.append(accuracy)
        
        # Stampa progresso
        print(f"  Run {run+1:2d}/{n_runs}: {accuracy:.2f}%")
        
        # Salva il modello migliore
        if run == 0 or accuracy == max(all_accuracies):
            best_net = net
    
    # Calcola statistiche
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    # Salva grafico del modello migliore
    best_net.save_plots("best_model_final.png")
    
    return mean_acc, std_acc, all_accuracies

def main():
    print("="*70)
    print("RICERCA DICOTOMICA - RETE NEURALE MONK DATASET")
    print("="*70)
    
    # Carico i dati MONK1 
    X_train_raw, y_train, X_test_raw, y_test = data.return_monk1()
    
    print(f"Forma dati grezzi:")
    print(f"  X_train: {X_train_raw.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test_raw.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    # Normalizzo tra -1 e 1 per compatibilità con tanh
    X_train = data.normalize(X_train_raw, -1, 1)
    X_test = data.normalize(X_test_raw, -1, 1)
    
    print(f"\n Dati dopo normalizzazione [-1, 1]:")
    print(f"  X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  X_test range:  [{X_test.min():.2f}, {X_test.max():.2f}]")
    
    # Inizializzo grid search dicotomica
    print(f"\n Configurazione ricerca dicotomica")
    grid_search = SimpleGridSearchCV(
        cv_folds=5,
        verbose=True,
        results_dir='dichotomic_results'
    )
    
    print(f"\n" + "="*70)
    print("AVVIO RICERCA DICOTOMICA")
    print("="*70)
    
    start_time = time.time()
    best_params, cv_mean, cv_std = grid_search.search_dichotomic(X_train, y_train)
    search_time = time.time() - start_time
    
    if best_params:
        print(f"\n Tempo ricerca: {search_time:.2f} secondi")
        
        
        print(f"\n" + "="*70)
        print("ALLENAMENTO FINALE SU TRAINING COMPLETO")
        print("="*70)
        
        print(f"\n📊 RISULTATI CROSS-VALIDATION (5-fold):")
        print(f"  Accuracy media:   {cv_mean:.2f}%")
        print(f"  Deviazione sandard:   ±{cv_std:.2f}%")
        print(f"  Intervallo 95%:   [{cv_mean - 1.96*cv_std:.2f}%, {cv_mean + 1.96*cv_std:.2f}%]")
        
       
        print(f"\n TRAINING FINALE E TEST:")
        test_mean, test_std, all_test_accs = train_final_model_multiple_runs(
            X_train, y_train, X_test, y_test, best_params, n_runs=10
        )
        
        # Calcola statistiche dettagliate
        test_min = np.min(all_test_accs)
        test_max = np.max(all_test_accs)
        test_median = np.median(all_test_accs)
        
        # RISULTATI FINALI
        print(f"\n" + "="*70)
        print("RISULTATI FINALI")
        print("="*70)
        
        print(f"\n STATISTICHE TEST SET (10 run):")
        print(f"  Media:        {test_mean:.2f}%")
        print(f"  Deviazione:   ±{test_std:.2f}%")
        print(f"  Minimo:       {test_min:.2f}%")
        print(f"  Massimo:      {test_max:.2f}%")
        print(f"  Intervallo:   [{test_mean - 1.96*test_std:.2f}%, {test_mean + 1.96*test_std:.2f}%]")
        
        # Salva risultati finali
        final_results = {
            'dataset': 'MONK1',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'search_time_seconds': float(search_time),
            'best_parameters': best_params,
            'cross_validation': {
                'accuracy_mean': float(cv_mean),
                'accuracy_std': float(cv_std),
                'confidence_95_lower': float(cv_mean - 1.96*cv_std),
                'confidence_95_upper': float(cv_mean + 1.96*cv_std)
            },
            'test_set_final': {
                'accuracy_mean': float(test_mean),
                'accuracy_std': float(test_std),
                'accuracy_median': float(test_median),
                'accuracy_min': float(test_min),
                'accuracy_max': float(test_max),
                'all_accuracies': [float(acc) for acc in all_test_accs],
                'confidence_95_lower': float(test_mean - 1.96*test_std),
                'confidence_95_upper': float(test_mean + 1.96*test_std)
            },
            'data_info': {
                'training_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'input_features': int(X_train.shape[1]),
                'normalization': 'range [-1, 1]'
            }
        }
        
        # Salva su file
        results_file = 'final_dichotomic_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n Risultati finali salvati in: {results_file}")
        
        return best_params, test_mean, test_std, all_test_accs
    
    return None, 0, 0, []

if __name__ == "__main__":
    print("\n INIZIO RICERCA DICOTOMICA")
    print("="*70)
    
    best_params, final_acc, final_std, all_accs = main()
    
    if best_params:
        print("\n" + "="*70)
        print("RIEPILOGO RICERCA")
        print("="*70)
        print(f"\n RISULTATO FINALE: {final_acc:.2f}% ± {final_std:.2f}%")
        print(f"\n PARAMETRI OTTIMALI:")
        for param, value in best_params.items():
            print(f"  {param:15s}: {value}")
        
        if all_accs:
            print(f"\n DISTRIBUZIONE ACCURACY (10 run):")
            for i, acc in enumerate(all_accs):
                print(f"  Run {i+1:2d}: {acc:6.2f}%")
        
        print(f"\n INTERPRETAZIONE:")
        print(f"  L'accuracy finale è {final_acc:.2f}% con un errore di ±{final_std:.2f}%")
        print(f"  Ciò significa che ci aspettiamo prestazioni tra")
        print(f"  {final_acc - 1.96*final_std:.2f}% e {final_acc + 1.96*final_std:.2f}% nel 95% dei casi")
    
    print("\n" + "="*70)
    print("RICERCA COMPLETATA")
    print("="*70)