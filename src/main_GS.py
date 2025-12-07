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
        n_trials=50,
        save_results=True,
        results_filename='trials_results.json'
    )
    
    print("\n" + "="*70)
    print("RISULTATO FINALE")
    print("="*70)
    print(f"Parametri ottimali: lr={best_params['learning_rate']:.3f}, "
          f"hidden={best_params['hidden_neurons']}, epochs={best_params['epochs']}")
    
    if trials_results:
        print(f"\nAccuracy su 50 trial: {trials_results['mean_accuracy']:.2f}% ± {trials_results['std_accuracy']:.2f}%")
        print(f"Range: {trials_results['min_accuracy']:.2f}% - {trials_results['max_accuracy']:.2f}%")

if __name__ == "__main__":
    main()