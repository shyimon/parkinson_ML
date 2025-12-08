import numpy as np
import time

# Import dei tuoi file
from data_manipulation import return_monk3
from grid_search_cv_M3 import GridSearchM3

def main():
    num_trials = 500
    print("="*70)
    print(f"GRID SEARCH + {num_trials} TRIAL SU MONK3")
    print("="*70)
    
    print("\n[1] Caricamento dataset MONK3")
    X_train, y_train, X_test, y_test = return_monk3(dataset_shuffle=True, one_hot=True)
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    print("\n[2] Esecuzione Grid Search Dicotomica...")
    print("-"*70)
    
    # Usa GridSearch ottimizzato per MONK3
    gs = GridSearchM3(cv_folds=5, verbose=True, results_dir='grid_search_results_monk3')
    
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
        n_trials=1000,
        save_results=True,
        results_filename='trials_results_monk3.json'
    )
    
    print("\n" + "="*70)
    print("RISULTATO FINALE - MONK3")
    print("="*70)
    print(f"Parametri ottimali: lr={best_params['learning_rate']:.4f}, "
          f"hidden={best_params['hidden_neurons']}, epochs={best_params['epochs']}")
    
    if trials_results:
        print(f"\nAccuracy su {num_trials} trial: {trials_results['mean_accuracy']:.2f}% ± {trials_results['std_accuracy']:.2f}%")
        print(f"Range: [{bes}] % - {trials_results['max_accuracy']:.2f}%")
    
    print("\n" + "="*70)
    print("NOTE: MONK3 contiene rumore (5% di etichette errate)")
    print("Range utilizzati:")
    print(f"  Learning rate: [0.005, 0.15]")
    print(f"  Hidden neurons: [3, 10]")
    print(f"  Epochs: [400, 1000]")
    print("="*70)

if __name__ == "__main__":
    main()