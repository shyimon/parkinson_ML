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
    
   
    best_params, cv_val_acc, cv_val_std, cv_train_acc, cv_train_std = gs._dichotomic_search(
        X_train, y_train,
        max_iteration=4,
        n_points=5
    )
    
    print(f"\n[3] {num_trials} Trial con parametri ottimali")
    print("-"*70)
    
    trials_results = gs.run_trials(
        X_train, y_train, X_test, y_test,
        n_trials=200,
        save_results=True,
        results_filename='trials_results.json'
    )
    
    print("\n" + "="*70)
    print("RISULTATO FINALE MONK 1")
    print("="*70)
    print(f"Parametri ottimali: lr={best_params['learning_rate']:.4f}, hidden={best_params['hidden_neurons']}, epochs={best_params['epochs']}, Training Accuracy:   {cv_train_acc:.2f}% ± {cv_train_std:.2f}%, Validation Accuracy: {cv_val_acc:.2f}% ± {cv_val_std:.2f}")
    
    
    if trials_results:
        print(f"\n{num_trials} Trial indipendenti:")
        print(f"  Training Accuracy: {trials_results['training_accuracy']['mean']:.2f}% ± {trials_results['training_accuracy']['std']:.2f}%")
        print(f"  Test Accuracy:     {trials_results['test_accuracy']['mean']:.2f}% ± {trials_results['test_accuracy']['std']:.2f}%")
        print(f"  Gap medio (Train-Test): {trials_results['accuracy_gap']['mean']:.2f}% ± {trials_results['accuracy_gap']['std']:.2f}%")
        print(f"  Tempo totale: {trials_results['time']['total']:.1f} secondi")

if __name__ == "__main__":
    main()