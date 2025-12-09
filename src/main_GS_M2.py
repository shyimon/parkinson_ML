import numpy as np
import time

# Import dei tuoi file
from data_manipulation import return_monk2
from grid_search_cv_M2 import GridSearchM2

def main():
    num_trials = 300
    print("="*70)
    print(f"GRID SEARCH + {num_trials} TRIAL SU MONK2")
    print("="*70)
    
    print("\n[1] Caricamento dataset MONK2...")
    X_train, y_train, X_test, y_test = return_monk2(dataset_shuffle=True, one_hot=True)
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    print("\n[2] Esecuzione Grid Search Dicotomica...")
    print("-"*70)
    
    # Usa GridSearch ottimizzato per MONK2
    gs = GridSearchM2(cv_folds=5, verbose=True, results_dir='grid_search_results_monk2')
    
    # I range sono già impostati nel costruttore per MONK2
    # gs.lr_range = [0.001, 0.1]
    # gs.hidden_range = [3, 8]
    # gs.epochs_range = [400, 1000]
    
    # Esegui ricerca dicotomica (con raffinamento finale)
    best_params, cv_acc, cv_std = gs._dichotomic_search(
        X_train, y_train,
        max_iteration=4,
        n_points=5
    )
    
    print(f"\n[3] {num_trials} Trial con parametri ottimali")
    print("-"*70)
    
    trials_results = gs.run_trials(
        X_train, y_train, X_test, y_test,
        n_trials=500,
        save_results=True,
        results_filename='trials_results_monk2.json'
    )
    
    print("\n" + "="*70)
    print("RISULTATO FINALE - MONK2")
    print("="*70)
    print(f"Parametri ottimali: lr={best_params['learning_rate']:.4f}, "
          f"hidden={best_params['hidden_neurons']}, epochs={best_params['epochs']}")
    
    if trials_results:
        print(f"\nAccuracy su {num_trials} trial: {trials_results['mean_accuracy']:.2f}% ± {trials_results['std_accuracy']:.2f}%")
        print(f"Range: {trials_results['min_accuracy']:.2f}% - {trials_results['max_accuracy']:.2f}%")
    
    print("\n" + "="*70)
    print("Range utilizzati:")
    print(f"  Learning rate: [0.001, 0.1]")
    print(f"  Hidden neurons: [3, 8]")
    print(f"  Epochs: [400, 1000]")
    print("="*70)

if __name__ == "__main__":
    main()