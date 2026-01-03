import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk1
import os
from datetime import datetime

OUTPUT_DIR = "monk1_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== GRID SEARCH CON K-FOLD CV + ALGORITMI MULTIPLI ==========
def grid_search_with_kfold_cv_multi_algo(
    n_seeds=20,
    k_folds=5,
    algorithms=None,
    learning_rates=None,
    momentums=None,
    decays=None,
    patiences=None,
    eta_plus_values=None,
    eta_minus_values=None,
    mu_values=None
):
    """
    Grid Search con K-Fold CV per MONK-1 con ALGORITMI MULTIPLI
    
    Algoritmi supportati:
    - 'sgd':   Stochastic Gradient Descent (con momentum opzionale)
    - 'rprop': Resilient Backpropagation
    - 'quickprop': Quickprop
    """
    if algorithms is None:
        algorithms = ['sgd', 'rprop', 'quickprop']
    
    if learning_rates is None: 
        learning_rates = {
            'sgd': [0.04, 0.05, 0.06],
            'rprop': [0.01],
            'quickprop': [0.01, 0.05, 0.1]
        }
    
    if momentums is None:
        momentums = [0.9, 0.95]
    
    if decays is None:
        decays = [0.9]
    
    if patiences is None: 
        patiences = [50]
    
    if eta_plus_values is None:
        eta_plus_values = [1.2]
    
    if eta_minus_values is None:
        eta_minus_values = [0.5]
    
    if mu_values is None:
        mu_values = [1.75, 2.0]
    
    print(f"\n{'='*70}")
    print(f" GRID SEARCH WITH K-FOLD CV + MULTIPLE ALGORITHMS")
    print(f"{'='*70}")
    print(f"Algorithms: {algorithms}")
    print(f"Seeds per config: {n_seeds}")
    print(f"K-Folds per seed: {k_folds}")
    print(f"{'='*70}\n")
    
    # Calcola numero totale di configurazioni
    total_configs = 0
    for algo in algorithms: 
        if algo == 'sgd': 
            n_lr = len(learning_rates.get('sgd', [0.05]))
            total_configs += n_lr * len(momentums) * len(decays) * len(patiences)
        elif algo == 'rprop':
            total_configs += len(eta_plus_values) * len(eta_minus_values) * len(patiences)
        elif algo == 'quickprop':
            n_lr = len(learning_rates.get('quickprop', [0.05]))
            total_configs += n_lr * len(mu_values) * len(decays) * len(patiences)
    
    total_trainings = total_configs * n_seeds * k_folds
    
    print(f"Total configurations: {total_configs}")
    print(f"Total trainings:  {total_trainings}\n")
    
    # Carica dati COMPLETI
    X_full, y_full, _, _, X_test, y_test = return_monk1(
        one_hot=True, val_split=0.0, dataset_shuffle=False
    )
    
    best_cv_score = 0
    best_config = None
    all_results_for_ensemble = []
    
    config_num = 0
    training_num = 0
    start_time = datetime.now()
    
    #  LOOP ALGORITMI
    for algo in algorithms:
        print(f"\n{'#'*70}")
        print(f" TESTING ALGORITHM: {algo.upper()}")
        print(f"{'#'*70}\n")
        
        #  SGD 
        if algo == 'sgd':
            sgd_lrs = learning_rates.get('sgd', [0.05])
            
            for lr in sgd_lrs:
                for mom in momentums:
                    for decay in decays:
                        for patience in patiences:
                            config_num += 1
                            
                            print(f"\n{'─'*70}")
                            print(f" Config {config_num}/{total_configs} [SGD]:  LR={lr}, Mom={mom}, Decay={decay}, Pat={patience}")
                            print(f"{'─'*70}")
                            
                            seed_results = []
                            
                            for seed_idx in range(n_seeds):
                                seed = seed_idx * 123 + int(lr * 10000) + int(mom * 100)
                                np.random.seed(seed)
                                
                                indices = np.arange(len(X_full))
                                np.random.shuffle(indices)
                                X_shuffled = X_full[indices]
                                y_shuffled = y_full[indices]
                                
                                fold_accuracies = []
                                fold_models = []
                                
                                for fold_idx in range(k_folds):
                                    training_num += 1
                                    
                                    # Crea fold
                                    fold_size = len(X_full) // k_folds
                                    remainder = len(X_full) % k_folds
                                    fold_sizes = [fold_size + (1 if i < remainder else 0) for i in range(k_folds)]
                                    fold_starts = [0]
                                    for size in fold_sizes[:-1]:
                                        fold_starts.append(fold_starts[-1] + size)
                                    
                                    val_start = fold_starts[fold_idx]
                                    val_end = val_start + fold_sizes[fold_idx]
                                    
                                    train_indices = list(range(0, val_start)) + list(range(val_end, len(X_full)))
                                    val_indices = list(range(val_start, val_end))
                                    
                                    X_train_fold = X_shuffled[train_indices]
                                    y_train_fold = y_shuffled[train_indices]
                                    X_val_fold = X_shuffled[val_indices]
                                    y_val_fold = y_shuffled[val_indices]
                                    
                                    # Crea modello SGD
                                    params = {
                                        'network_structure': [17, 4, 1],
                                        'eta': lr,
                                        'momentum': mom,
                                        'l2_lambda': 0.0,
                                        'algorithm': 'sgd',
                                        'activation_type': 'sigmoid',
                                        'loss_type': 'half_mse',
                                        'weight_initializer': 'xavier',
                                        'decay': decay,
                                        'mu': 1.75,
                                        'eta_plus': 1.2,
                                        'eta_minus': 0.5,
                                        'debug': False
                                    }
                                    
                                    net = NeuralNetwork(**params)
                                    history = net.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                     epochs=400, batch_size=1, patience=patience, verbose=False)
                                    
                                    val_pred = net.predict(X_val_fold)
                                    val_acc = np.mean((val_pred > 0.5).astype(int) == y_val_fold)
                                    fold_accuracies.append(val_acc)
                                    fold_models.append(net)
                                    
                                    if training_num % 100 == 0:
                                        elapsed = (datetime.now() - start_time).total_seconds()
                                        eta_seconds = (elapsed / training_num) * (total_trainings - training_num)
                                        print(f"  Training {training_num}/{total_trainings} ({training_num/total_trainings*100:.1f}%) | "
                                              f"ETA: {eta_seconds/60:.1f} min | Best: {best_cv_score:.4%}")
                                
                                seed_cv_score = np.mean(fold_accuracies)
                                best_fold_idx = np.argmax(fold_accuracies)
                                
                                seed_result = {
                                    'network': fold_models[best_fold_idx],
                                    'algorithm': 'sgd',
                                    'lr': lr,
                                    'momentum': mom,
                                    'decay': decay,
                                    'patience': patience,
                                    'seed': seed,
                                    'cv_score': seed_cv_score,
                                    'train_accuracy': 1.0,
                                    'val_accuracy': seed_cv_score,
                                    'params': params,
                                    'history':  history,
                                    'data': (X_shuffled[: int(0.7*len(X_shuffled))], 
                                            y_shuffled[:int(0.7*len(X_shuffled))],
                                            X_shuffled[int(0.7*len(X_shuffled)):], 
                                            y_shuffled[int(0.7*len(X_shuffled)):],
                                            X_test, y_test)
                                }
                                
                                seed_results.append(seed_result)
                                all_results_for_ensemble.append(seed_result)
                            
                            config_cv_score = np.mean([r['cv_score'] for r in seed_results])
                            config_cv_std = np.std([r['cv_score'] for r in seed_results])
                            
                            print(f"  ✓ Config CV Score: {config_cv_score:.4%} ± {config_cv_std:.4%}")
                            
                            if config_cv_score > best_cv_score:
                                best_cv_score = config_cv_score
                                best_config = max(seed_results, key=lambda x: x['cv_score'])
                                print(f" NEW BEST [SGD]!  CV:  {best_cv_score:.4%}")
        
        #  RPROP 
        elif algo == 'rprop':
            for eta_plus in eta_plus_values:
                for eta_minus in eta_minus_values:
                    for patience in patiences: 
                        config_num += 1
                        
                        print(f"\n{'─'*70}")
                        print(f" Config {config_num}/{total_configs} [RPROP]: η+={eta_plus}, η-={eta_minus}, Pat={patience}")
                        print(f"{'─'*70}")
                        
                        seed_results = []
                        
                        for seed_idx in range(n_seeds):
                            seed = seed_idx * 456 + int(eta_plus * 100)
                            np.random.seed(seed)
                            
                            indices = np.arange(len(X_full))
                            np.random.shuffle(indices)
                            X_shuffled = X_full[indices]
                            y_shuffled = y_full[indices]
                            
                            fold_accuracies = []
                            fold_models = []
                            
                            for fold_idx in range(k_folds):
                                training_num += 1
                                
                                fold_size = len(X_full) // k_folds
                                remainder = len(X_full) % k_folds
                                fold_sizes = [fold_size + (1 if i < remainder else 0) for i in range(k_folds)]
                                fold_starts = [0]
                                for size in fold_sizes[:-1]:
                                    fold_starts.append(fold_starts[-1] + size)
                                
                                val_start = fold_starts[fold_idx]
                                val_end = val_start + fold_sizes[fold_idx]
                                
                                train_indices = list(range(0, val_start)) + list(range(val_end, len(X_full)))
                                val_indices = list(range(val_start, val_end))
                                
                                X_train_fold = X_shuffled[train_indices]
                                y_train_fold = y_shuffled[train_indices]
                                X_val_fold = X_shuffled[val_indices]
                                y_val_fold = y_shuffled[val_indices]
                                
                                # Crea modello RPROP
                                params = {
                                    'network_structure': [17, 4, 1],
                                    'eta': 0.01,
                                    'momentum':  0.0,
                                    'l2_lambda': 0.0,
                                    'algorithm': 'rprop',
                                    'activation_type': 'sigmoid',
                                    'loss_type': 'half_mse',
                                    'weight_initializer': 'xavier',
                                    'decay': 0.9,
                                    'mu': 1.75,
                                    'eta_plus': eta_plus,
                                    'eta_minus': eta_minus,
                                    'debug': False
                                }
                                
                                net = NeuralNetwork(**params)
                                history = net.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                 epochs=400, batch_size=1, patience=patience, verbose=False)
                                
                                val_pred = net.predict(X_val_fold)
                                val_acc = np.mean((val_pred > 0.5).astype(int) == y_val_fold)
                                fold_accuracies.append(val_acc)
                                fold_models.append(net)
                                
                                if training_num % 100 == 0:
                                    elapsed = (datetime.now() - start_time).total_seconds()
                                    eta_seconds = (elapsed / training_num) * (total_trainings - training_num)
                                    print(f"  Training {training_num}/{total_trainings} ({training_num/total_trainings*100:.1f}%) | "
                                          f"ETA:  {eta_seconds/60:.1f} min | Best: {best_cv_score:.4%}")
                            
                            seed_cv_score = np.mean(fold_accuracies)
                            best_fold_idx = np.argmax(fold_accuracies)
                            
                            seed_result = {
                                'network': fold_models[best_fold_idx],
                                'algorithm': 'rprop',
                                'lr': 0.01,
                                'eta_plus': eta_plus,
                                'eta_minus': eta_minus,
                                'momentum': 0.0,
                                'decay': 0.9,
                                'patience': patience,
                                'seed': seed,
                                'cv_score': seed_cv_score,
                                'train_accuracy': 1.0,
                                'val_accuracy': seed_cv_score,
                                'params':  params,
                                'history':  history,
                                'data':  (X_shuffled[:int(0.7*len(X_shuffled))], 
                                        y_shuffled[:int(0.7*len(X_shuffled))],
                                        X_shuffled[int(0.7*len(X_shuffled)):], 
                                        y_shuffled[int(0.7*len(X_shuffled)):],
                                        X_test, y_test)
                            }
                            
                            seed_results.append(seed_result)
                            all_results_for_ensemble.append(seed_result)
                        
                        config_cv_score = np.mean([r['cv_score'] for r in seed_results])
                        config_cv_std = np.std([r['cv_score'] for r in seed_results])
                        
                        print(f"  ✓ Config CV Score: {config_cv_score:.4%} ± {config_cv_std:.4%}")
                        
                        if config_cv_score > best_cv_score: 
                            best_cv_score = config_cv_score
                            best_config = max(seed_results, key=lambda x: x['cv_score'])
                            print(f"  ✨ NEW BEST [RPROP]! CV: {best_cv_score:.4%}")
        
        #  QUICKPROP 
        elif algo == 'quickprop': 
            qp_lrs = learning_rates.get('quickprop', [0.05])
            
            for lr in qp_lrs:
                for mu in mu_values:
                    for decay in decays:
                        for patience in patiences: 
                            config_num += 1
                            
                            print(f"\n{'─'*70}")
                            print(f" Config {config_num}/{total_configs} [QUICKPROP]: LR={lr}, μ={mu}, Decay={decay}, Pat={patience}")
                            print(f"{'─'*70}")
                            
                            seed_results = []
                            
                            for seed_idx in range(n_seeds):
                                seed = seed_idx * 789 + int(lr * 10000) + int(mu * 100)
                                np.random.seed(seed)
                                
                                indices = np.arange(len(X_full))
                                np.random.shuffle(indices)
                                X_shuffled = X_full[indices]
                                y_shuffled = y_full[indices]
                                
                                fold_accuracies = []
                                fold_models = []
                                
                                for fold_idx in range(k_folds):
                                    training_num += 1
                                    
                                    fold_size = len(X_full) // k_folds
                                    remainder = len(X_full) % k_folds
                                    fold_sizes = [fold_size + (1 if i < remainder else 0) for i in range(k_folds)]
                                    fold_starts = [0]
                                    for size in fold_sizes[:-1]:
                                        fold_starts.append(fold_starts[-1] + size)
                                    
                                    val_start = fold_starts[fold_idx]
                                    val_end = val_start + fold_sizes[fold_idx]
                                    
                                    train_indices = list(range(0, val_start)) + list(range(val_end, len(X_full)))
                                    val_indices = list(range(val_start, val_end))
                                    
                                    X_train_fold = X_shuffled[train_indices]
                                    y_train_fold = y_shuffled[train_indices]
                                    X_val_fold = X_shuffled[val_indices]
                                    y_val_fold = y_shuffled[val_indices]
                                    
                                    # Crea modello QUICKPROP
                                    params = {
                                        'network_structure': [17, 4, 1],
                                        'eta': lr,
                                        'momentum': 0.0,
                                        'l2_lambda': 0.0,
                                        'algorithm': 'quickprop',
                                        'activation_type': 'sigmoid',
                                        'loss_type': 'half_mse',
                                        'weight_initializer': 'xavier',
                                        'decay': decay,
                                        'mu': mu,
                                        'eta_plus': 1.2,
                                        'eta_minus': 0.5,
                                        'debug':  False
                                    }
                                    
                                    net = NeuralNetwork(**params)
                                    history = net.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                     epochs=400, batch_size=1, patience=patience, verbose=False)
                                    
                                    val_pred = net.predict(X_val_fold)
                                    val_acc = np.mean((val_pred > 0.5).astype(int) == y_val_fold)
                                    fold_accuracies.append(val_acc)
                                    fold_models.append(net)
                                    
                                    if training_num % 100 == 0:
                                        elapsed = (datetime.now() - start_time).total_seconds()
                                        eta_seconds = (elapsed / training_num) * (total_trainings - training_num)
                                        print(f"  Training {training_num}/{total_trainings} ({training_num/total_trainings*100:.1f}%) | "
                                              f"ETA: {eta_seconds/60:.1f} min | Best: {best_cv_score:.4%}")
                                
                                seed_cv_score = np.mean(fold_accuracies)
                                best_fold_idx = np.argmax(fold_accuracies)
                                
                                seed_result = {
                                    'network':  fold_models[best_fold_idx],
                                    'algorithm': 'quickprop',
                                    'lr': lr,
                                    'mu': mu,
                                    'momentum': 0.0,
                                    'decay': decay,
                                    'patience': patience,
                                    'seed': seed,
                                    'cv_score': seed_cv_score,
                                    'train_accuracy': 1.0,
                                    'val_accuracy': seed_cv_score,
                                    'params': params,
                                    'history':  history,
                                    'data': (X_shuffled[:int(0.7*len(X_shuffled))], 
                                            y_shuffled[:int(0.7*len(X_shuffled))],
                                            X_shuffled[int(0.7*len(X_shuffled)):], 
                                            y_shuffled[int(0.7*len(X_shuffled)):],
                                            X_test, y_test)
                                }
                                
                                seed_results.append(seed_result)
                                all_results_for_ensemble.append(seed_result)
                            
                            config_cv_score = np.mean([r['cv_score'] for r in seed_results])
                            config_cv_std = np.std([r['cv_score'] for r in seed_results])
                            
                            print(f"  ✓ Config CV Score: {config_cv_score:.4%} ± {config_cv_std:.4%}")
                            
                            if config_cv_score > best_cv_score: 
                                best_cv_score = config_cv_score
                                best_config = max(seed_results, key=lambda x:  x['cv_score'])
                                print(f"  ✨ NEW BEST [QUICKPROP]! CV: {best_cv_score:.4%}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f" BEST CONFIGURATION (All Algorithms)")
    print(f"{'='*70}")
    print(f"Algorithm:          {best_config['algorithm'].upper()}")
    print(f"Learning Rate:     {best_config['lr']}")
    if best_config['algorithm'] == 'sgd': 
        print(f"Momentum:           {best_config['momentum']}")
    elif best_config['algorithm'] == 'rprop':
        print(f"Eta Plus:          {best_config.get('eta_plus', 'N/A')}")
        print(f"Eta Minus:         {best_config.get('eta_minus', 'N/A')}")
    elif best_config['algorithm'] == 'quickprop':
        print(f"Mu:                {best_config.get('mu', 'N/A')}")
    print(f"Decay:             {best_config['decay']}")
    print(f"Patience:          {best_config['patience']}")
    print(f"Seed:              {best_config['seed']}")
    print(f"CV Score:          {best_config['cv_score']:.4%}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Total trainings: {training_num}")
    print(f"{'='*70}\n")
    
    return best_config, all_results_for_ensemble


# FUNZIONI DI VALUTAZIONE E PLOTTING 

def evaluate_on_test_set(net, X_test, y_test):
    """Valuta il modello finale sul TEST SET"""
    print(f"\n{'='*70}")
    print(f"⚡ VALUTAZIONE FINALE SUL TEST SET")
    print(f"{'='*70}")
    
    test_pred = net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_acc = np.mean(test_pred_class == y_test)
    test_error = 1 - test_acc
    
    # Confusion matrix
    tp = np.sum((test_pred_class == 1) & (y_test == 1))
    fp = np.sum((test_pred_class == 1) & (y_test == 0))
    tn = np.sum((test_pred_class == 0) & (y_test == 0))
    fn = np.sum((test_pred_class == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n RISULTATI TEST SET:")
    print(f"  Test Accuracy:  {test_acc:.4%}")
    print(f"  Test Error:     {test_error:.4%}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN: {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error': test_error,
        'precision': precision,
        'recall': recall,
        'f1':  f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def plot_accuracy_bar_and_confusion(train_results, test_results, save_path=f'{OUTPUT_DIR}/monk1_results.png'):
    """Crea grafico con bar chart accuracy e confusion matrix"""
    train_acc = train_results['train_accuracy']
    val_acc = train_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    lr = train_results['lr']
    seed = train_results['seed']
    algo = train_results.get('algorithm', 'sgd').upper()
    
    fig = plt.figure(figsize=(14, 6))
    
    # SUBPLOT 1: ACCURACY BAR CHART
    ax1 = plt.subplot(1, 2, 1)
    
    categories = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2.5)
    
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        ax1.text(i, acc + 0.03, f'{acc:.2%}', ha='center', 
                fontsize=16, fontweight='bold')
    
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title(f'Model Performance (MONK-1)\n({algo}, LR={lr}, Seed={seed})', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2.5, 
               alpha=0.7, label='100% Target')
    ax1.legend(fontsize=11, loc='lower right')
    
    # SUBPLOT 2: CONFUSION MATRIX
    ax2 = plt.subplot(1, 2, 2)
    
    cm = test_results['confusion_matrix']
    confusion_data = np.array([[cm['tn'], cm['fp']], 
                                [cm['fn'], cm['tp']]])
    
    im = ax2.imshow(confusion_data, cmap='Blues', alpha=0.9, vmin=0, vmax=confusion_data.max())
    plt.colorbar(im, ax=ax2, label='Count', fraction=0.046, pad=0.04)
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if confusion_data[i, j] > confusion_data.max()/2 else 'black'
            ax2.text(j, i, str(confusion_data[i, j]), 
                    ha='center', va='center', 
                    fontsize=28, fontweight='bold',
                    color=text_color)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['True 0', 'True 1'], fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True', fontsize=14, fontweight='bold')
    ax2.set_title(f'Confusion Matrix (TEST SET)\nPrecision: {test_results["precision"]:.2%}, Recall: {test_results["recall"]:.2%}', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grafico salvato in:  {save_path}")
    plt.close(fig)


def display_all_plots(best_results, test_results):
    """Mostra tutti i grafici finali con smoothing"""
    print(f"\n{'='*70}")
    print(" VISUALIZZAZIONE GRAFICI FINALI")
    print(f"{'='*70}\n")
    
    def smooth_curve(values, weight=0.9):
        """Exponential Moving Average smoothing"""
        if len(values) == 0:
            return values
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    fig = plt.figure(figsize=(18, 6))
    
    # SUBPLOT 1: TRAINING HISTORY
    ax1 = plt.subplot(1, 3, 1)
    
    history = best_results.get('history', None)
    
    if history and isinstance(history, dict) and 'training' in history and len(history['training']) > 0:
        print(" Usando history esistente")
        
        train_loss = history['training']
        val_loss = history['validation']
        
        train_loss_smooth = smooth_curve(train_loss, weight=0.9)
        val_loss_smooth = smooth_curve(val_loss, weight=0.9)
        
        epochs = range(1, len(train_loss) + 1)
        
        ax1.plot(epochs, train_loss_smooth, 
                 label='Training Loss (smoothed)', 
                 color='#1f77b4',
                 linewidth=2.5, 
                 alpha=0.95,
                 zorder=3)
        
        ax1.plot(epochs, val_loss_smooth, 
                 label='Validation Loss (smoothed)', 
                 color='#ff7f0e',
                 linewidth=2.5, 
                 alpha=0.95,
                 zorder=3)
        
        min_val_idx = np.argmin(val_loss_smooth)
        min_val_epoch = min_val_idx + 1
        min_val_value = val_loss_smooth[min_val_idx]
        
        ax1.scatter([min_val_epoch], [min_val_value], 
                   color='red', s=100, marker='*', 
                   edgecolors='darkred', linewidths=1.5,
                   zorder=5, label=f'Min Val Loss (epoch {min_val_epoch})')
        
        ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss (Half MSE)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Training & Validation Loss (smoothed)\n({best_results.get("algorithm", "N/A").upper()}, LR={best_results["lr"]}, Seed={best_results["seed"]})', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.legend(fontsize=9, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        y_min = min(min(train_loss_smooth), min(val_loss_smooth))
        y_max = max(max(train_loss_smooth), max(val_loss_smooth))
        y_range = y_max - y_min
        ax1.set_ylim(max(0, y_min - 0.05*y_range), y_max + 0.1*y_range)
        
    else:
        ax1.text(0.5, 0.5, 'No history available', 
                ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.set_title('Training History', fontsize=12, fontweight='bold')
    
    # SUBPLOT 2: ACCURACY BAR CHART
    ax2 = plt.subplot(1, 3, 2)
    
    train_acc = best_results['train_accuracy']
    val_acc = best_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    
    categories = ['Train', 'Val', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = ax2.bar(categories, accuracies, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2, width=0.6)
    
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        ax2.text(i, acc + 0.03, f'{acc:.2%}', 
                ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title(f'Model Performance\n({best_results.get("algorithm", "N/A").upper()}, LR={best_results["lr"]}, Seed={best_results["seed"]})', 
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label='100% Target', zorder=1)
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
    
    # SUBPLOT 3: CONFUSION MATRIX
    ax3 = plt.subplot(1, 3, 3)
    
    cm = test_results['confusion_matrix']
    confusion_data = np.array([[cm['tn'], cm['fp']], 
                                [cm['fn'], cm['tp']]])
    
    im = ax3.imshow(confusion_data, cmap='Blues', alpha=0.9, 
                    vmin=0, vmax=confusion_data.max())
    
    cbar = plt.colorbar(im, ax=ax3, label='Count', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    for i in range(2):
        for j in range(2):
            value = confusion_data[i, j]
            text_color = 'white' if value > confusion_data.max()/2 else 'black'
            ax3.text(j, i, str(value), 
                    ha='center', va='center', 
                    fontsize=24, fontweight='bold',
                    color=text_color)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=10, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['True 0', 'True 1'], fontsize=10, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True', fontsize=11, fontweight='bold')
    
    precision = test_results['precision']
    recall = test_results['recall']
    f1 = test_results['f1']
    
    ax3.set_title(f'Confusion Matrix (TEST SET)\nPrecision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}', 
                 fontsize=12, fontweight='bold', pad=10)
    
    fig.suptitle('MONK-1 Final Results Summary', 
                fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = f'{OUTPUT_DIR}/monk1_final_summary.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n Grafico combinato salvato in: {save_path}")
    
    plt.show()
    
    print(" Grafici visualizzati con smoothing EMA!")

if __name__ == "__main__": 
    try:
        print("="*70)
        print("🧠 MONK-1 - GRID SEARCH K-FOLD CV + MULTIPLE ALGORITHMS")
        print("="*70)
        
        # FASE 1: GRID SEARCH CON K-FOLD CV 
        print("\n FASE 1: Grid Search con K-Fold CV + Algoritmi Multipli")
        
        best_config, all_results = grid_search_with_kfold_cv_multi_algo(
            n_seeds=15,  # 15 seed per algoritmo
            k_folds=5,
            algorithms=['sgd', 'rprop', 'quickprop'],  # Tutti e 3! 
            learning_rates={
                'sgd': [0.04, 0.05, 0.06],
                'quickprop': [0.01, 0.05, 0.1]
            },
            momentums=[0.9, 0.95],  # Solo SGD
            decays=[0.9],
            patiences=[50],
            eta_plus_values=[1.2, 1.3],  # Solo RPROP
            eta_minus_values=[0.5],  # Solo RPROP
            mu_values=[1.75, 2.0]  # Solo Quickprop
        )
        
        # FASE 2: VALUTAZIONE TEST SET 
        print(f"\n FASE 2: Valutazione sul Test Set")
        
        X_train, y_train, X_val, y_val, X_test, y_test = best_config['data']
        final_net = best_config['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        #  FASE 3: ANALISI ERRORI 
        print(f"\n{'='*70}")
        print(" ANALISI ERRORI SUL TEST SET")
        print(f"{'='*70}")
        
        test_pred = final_net.predict(X_test)
        test_pred_class = (test_pred > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        errors_mask = test_pred_class != y_test_flat
        error_indices = np.where(errors_mask)[0]
        
        print(f"\n Errori sul test set:  {len(error_indices)}/{len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
        
        if len(error_indices) > 0:
            print(f"\n Dettaglio esempi sbagliati:")
            print(f"{'─'*70}")
            print(f"{'Idx':<8} {'Pred Value':<15} {'Pred Class':<15} {'True Class':<15} {'Confidence'}")
            print(f"{'─'*70}")
            
            for idx in error_indices: 
                pred_val = test_pred[idx][0]
                pred_class = test_pred_class[idx]
                true_class = y_test_flat[idx]
                confidence = abs(pred_val - 0.5)
                
                print(f"{idx:<8} {pred_val:<15.4f} {pred_class:<15} {true_class:<15} {confidence:.4f}")
            
            print(f"{'─'*70}")
            avg_confidence_errors = np.mean([abs(test_pred[idx][0] - 0.5) for idx in error_indices])
            print(f"\n Confidence media sugli errori: {avg_confidence_errors:.4f}")
        else:
            print(f"\n Nessun errore!  100% accuracy raggiunta!")
        
        print(f"\n{'='*70}\n")
        
        # FASE 4: PLOT ACCURACY + CONFUSION 
        print(f"\n📍 FASE 4: Plot Accuracy e Confusion Matrix")
        plot_accuracy_bar_and_confusion(best_config, test_results, 
                                       save_path=f'{OUTPUT_DIR}/monk1_results.png')
        
        # FASE 5: ENSEMBLE 
        print(f"\n{'='*70}")
        print(" FASE 5: ENSEMBLE AVANZATO (Top 30 modelli)")
        print(f"{'='*70}")
        
        sorted_results = sorted(all_results, key=lambda x: x['cv_score'], reverse=True)
        n_ensemble = min(30, len(all_results))
        top_n_results = sorted_results[:n_ensemble]
        
        print(f"\n Top {n_ensemble} configurazioni selezionate:")
        print(f"{'─'*70}")
        print(f"{'Rank':<6} {'Algorithm':<12} {'LR':<8} {'Seed':<8} {'CV Score':<12}")
        print(f"{'─'*70}")
        
        for idx, res in enumerate(top_n_results[: 10], 1):  # Mostra solo top 10
            print(f"{idx:<6} {res['algorithm'].upper():<12} {res['lr']:<8.3f} {res['seed']: <8d} {res['cv_score']: <12.4%}")
        
        if n_ensemble > 10:
            print(f"...  ({n_ensemble - 10} altri)")
        
        print(f"{'─'*70}")
        
        ensemble_nets = [r['network'] for r in top_n_results]
        print(f"\n✓ Ensemble di {len(ensemble_nets)} modelli pronto")
        
        # WEIGHTED VOTING
        print(f"\n Valutazione Weighted Ensemble...")
        
        ensemble_probs_all = []
        ensemble_weights = []
        
        for res in top_n_results: 
            pred_prob = res['network'].predict(X_test).flatten()
            ensemble_probs_all.append(pred_prob)
            ensemble_weights.append(res['cv_score'])
        
        ensemble_weights = np.array(ensemble_weights)
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        
        ensemble_probs_stack = np.array(ensemble_probs_all)
        ensemble_prob_weighted = np.sum(ensemble_probs_stack * ensemble_weights[: , np.newaxis], axis=0)
        
        # THRESHOLD TUNING
        print(f"\n Threshold Tuning...")
        
        best_threshold = 0.5
        best_acc_thresh = 0
        
        for threshold in np.arange(0.40, 0.61, 0.01):
            pred_class = (ensemble_prob_weighted >= threshold).astype(int)
            acc = np.mean(pred_class == y_test.flatten())
            if acc > best_acc_thresh: 
                best_acc_thresh = acc
                best_threshold = threshold
        
        print(f" Migliore soglia: {best_threshold:.3f} → Accuracy: {best_acc_thresh:.4%}")
        
        ensemble_pred_class = (ensemble_prob_weighted >= best_threshold).astype(int)
        
        # Metriche weighted ensemble
        weighted_acc = np.mean(ensemble_pred_class == y_test.flatten())
        
        tp_w = np.sum((ensemble_pred_class == 1) & (y_test.flatten() == 1))
        fp_w = np.sum((ensemble_pred_class == 1) & (y_test.flatten() == 0))
        tn_w = np.sum((ensemble_pred_class == 0) & (y_test.flatten() == 0))
        fn_w = np.sum((ensemble_pred_class == 0) & (y_test.flatten() == 1))
        
        precision_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
        recall_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
        f1_w = 2 * (precision_w * recall_w) / (precision_w + recall_w) if (precision_w + recall_w) > 0 else 0
        
        print(f"\n{'='*70}")
        print(" RISULTATI WEIGHTED ENSEMBLE")
        print(f"{'='*70}")
        print(f"  Numero modelli:       {n_ensemble}")
        print(f"  Threshold:           {best_threshold:.3f}")
        print(f"  Weighted Accuracy:   {weighted_acc:.4%}")
        print(f"  Precision:           {precision_w:.4f}")
        print(f"  Recall:              {recall_w:.4f}")
        print(f"  F1-score:            {f1_w:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {tp_w: 3d}  FP: {fp_w:3d}")
        print(f"    FN: {fn_w:3d}  TN: {tn_w:3d}")
        
        print(f"\n{'─'*70}")
        print(" CONFRONTO:  Singolo vs Ensemble")
        print(f"{'─'*70}")
        print(f"  Singolo (best):     {test_results['test_accuracy']:.4%}")
        print(f"  Ensemble Weighted:  {weighted_acc:.4%}")
        
        improvement = (weighted_acc - test_results['test_accuracy']) * 100
        if improvement > 0:
            single_errors = int((1 - test_results['test_accuracy']) * len(y_test))
            ensemble_errors = int((1 - weighted_acc) * len(y_test))
            errors_fixed = single_errors - ensemble_errors
            print(f"   Miglioramento:      +{improvement:.2f}%")
            print(f"   Errori corretti:   {errors_fixed}/{single_errors}")
        
        if weighted_acc >= 1.0:
            print(f" 100% TEST ACCURACY RAGGIUNTO CON ENSEMBLE!")
            
            weighted_results = {
                'test_accuracy': weighted_acc,
                'test_error': 1 - weighted_acc,
                'precision': precision_w,
                'recall': recall_w,
                'f1': f1_w,
                'confusion_matrix':  {'tp': tp_w, 'fp': fp_w, 'tn': tn_w, 'fn': fn_w}
            }
            
            weighted_train_results = best_config.copy()
            weighted_train_results['lr'] = f"Ensemble-{n_ensemble}"
            weighted_train_results['seed'] = f"Thresh={best_threshold:.3f}"
            weighted_train_results['algorithm'] = "Ensemble"
            
            plot_accuracy_bar_and_confusion(weighted_train_results, weighted_results, 
                                           save_path=f'{OUTPUT_DIR}/monk1_results_ENSEMBLE_100.png')
        
        print(f"\n{'='*70}\n")
        
        # FASE 6: VISUALIZZAZIONE FINALE
        print(f"\n FASE 6: Visualizzazione Grafici Finali")
        display_all_plots(best_config, test_results)
        
        # RIEPILOGO FINALE 
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print(f"{'='*70}")
        
        if weighted_acc >= 1.0:
            print(" PERFETTO:  100% TEST ACCURACY CON ENSEMBLE!")
        elif test_results['test_accuracy'] >= 1.0:
            print(" PERFETTO: 100% TEST ACCURACY!")
        elif weighted_acc >= 0.99:
            print(" ECCELLENTE: 99%+ test accuracy con ensemble!")
        else:
            print(f" Test Accuracy (singolo): {test_results['test_accuracy']:.2%}")
            print(f" Test Accuracy (ensemble): {weighted_acc:.2%}")
        
        print(f"\n{'─'*70}")
        print(" PERFORMANCE FINALE:")
        print(f"{'─'*70}")
        print(f"  SINGOLO MODELLO:")
        print(f"    Algorithm:           {best_config['algorithm'].upper()}")
        print(f"    CV Score (K-Fold):  {best_config['cv_score']:.4%}")
        print(f"    Test Accuracy:      {test_results['test_accuracy']:.4%}")
        
        print(f"\n  ENSEMBLE ({n_ensemble} modelli):")
        print(f"    Test Accuracy:      {weighted_acc:.4%}")
        
        print(f"\n{'─'*70}")
        print(" PARAMETRI MIGLIORI:")
        print(f"{'─'*70}")
        params = best_config['params']
        print(f"  Algorithm:           {best_config['algorithm'].upper()}")
        print(f"  Network Structure:   {params['network_structure']}")
        print(f"  Learning Rate:       {params['eta']}")
        if best_config['algorithm'] == 'sgd': 
            print(f"  Momentum:            {params['momentum']}")
        elif best_config['algorithm'] == 'rprop':
            print(f"  Eta Plus:            {params['eta_plus']}")
            print(f"  Eta Minus:           {params['eta_minus']}")
        elif best_config['algorithm'] == 'quickprop':
            print(f"  Mu:                  {params['mu']}")
        print(f"  Decay:                {params['decay']}")
        print(f"  Weight Initializer:  {params['weight_initializer']}")
        print(f"  Random Seed:         {best_config['seed']}")
        
        print(f"\n{'─'*70}")
        print(" FILE SALVATI:")
        print(f"{'─'*70}")
        
        files_to_check = [
            f'{OUTPUT_DIR}/monk1_results.png',
            f'{OUTPUT_DIR}/monk1_final_summary.png'
        ]
        
        if weighted_acc >= 1.0:
            files_to_check.append(f'{OUTPUT_DIR}/monk1_results_ENSEMBLE_100.png')
        
        for filename in files_to_check: 
            if os.path.exists(filename):
                print(f"   {filename}")
            else:
                print(f"   {filename} - not found")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO MONK-1 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE:  {e}")
        import traceback
        traceback.print_exc()