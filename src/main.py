import numpy as np
import data_manipulation as data
import neural_network as nn
from grid_search import GridSearch

def main():
    print("\n[1] Caricamento dataset MONK3...")
    X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk3(one_hot=True, dataset_shuffle=True)

    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    print("\n[2] Esecuzione Grid Search...")
    print("-"*70)

    # Normalization
    # X_train_normalized = data.normalize(X_train, 0, 1, X_train.min(axis=0), X_train.max(axis=0))
    # X_test_normalized = data.normalize(X_test, 0, 1, X_train.min(axis=0), X_train.max(axis=0))
    X_train_normalized = np.concatenate(X_train, X_val)
    y_test = np.concatenate(y_train, y_val)
    X_test_normalized = X_test

    gs = GridSearch(cv_folds=5, verbose=True, results_dir='grid_search_results')
   
    best_params, cv_val_acc, cv_val_std, cv_train_acc, cv_train_std = gs._dichotomic_search(
        X_train_normalized, y_train,
        max_iteration=4,
        n_points=5
    )

    num_trials = 500
    print("="*70)
    print(f"GRID SEARCH + {num_trials} TRIAL SU MONK1")
    print("="*70)

    print(f"\n[3] {num_trials} Trial con parametri ottimali")
    print("-"*70)
    
    trials_results = gs.run_trials(
        X_train_normalized, y_train, X_test_normalized, y_test,
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

    # net = nn.NeuralNetwork(network_structure, eta=eta, loss_type="half_mse", l2_lambda=0.0001, algorithm="sgd", activation_type="sigmoid", eta_plus=1.2, eta_minus=0.5, mu=1.75, decay=0.0001, weight_initializer="def", momentum=0.9)
    # net.fit(X_train_normalized, y_train, X_val_normalized, y_val, epochs=2000, batch_size=4, patience=50)
if __name__ == "__main__":
    main()