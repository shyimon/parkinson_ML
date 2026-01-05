import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_CUP, normalize, MEE, MSE, denormalize
import warnings

# Filtra i RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)


def smooth_curve(values, weight=0.9):
    """
    Applica exponential moving average per smoothing
    """
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def create_k_folds(X, y, k=5, shuffle=True, seed=42):
    """
    Divide i dati in k folds per cross validation
    
    Args:
        X: features
        y: targets
        k: numero di folds
        shuffle: se mescolare i dati prima di dividere
        seed: random seed per riproducibilità
    
    Returns:
        Lista di tuple (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    
    folds = []
    current = 0
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        current = stop
    
    return folds


def train_single_fold(X_train, y_train, X_val, y_val, params, verbose=False):
    """
    Addestra su un singolo fold
    
    Returns:
        dict con train_mee, val_mee, train_mse, val_mse
    """
    # Normalizzazione input (0-1)
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    X_train_norm = normalize(X_train, 0, 1, x_min, x_max)
    X_val_norm = normalize(X_val, 0, 1, x_min, x_max)
    
    # Normalizzazione target (-1, 1) per tanh
    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)
    y_train_norm = normalize(y_train, -1, 1, y_min, y_max)
    y_val_norm = normalize(y_val, -1, 1, y_min, y_max)
    
    # Crea e addestra rete
    net = NeuralNetwork(**params)
    
    history = net.fit(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        epochs=2000,
        batch_size=16,
        patience=300,
        verbose=verbose
    )
    
    # Predizioni
    train_pred_norm = net.predict(X_train_norm)
    val_pred_norm = net.predict(X_val_norm)
    
    # Denormalizza
    train_pred = denormalize(train_pred_norm, -1, 1, y_min, y_max)
    val_pred = denormalize(val_pred_norm, -1, 1, y_min, y_max)
    
    # Calcola metriche
    train_mee = MEE(y_train, train_pred)
    val_mee = MEE(y_val, val_pred)
    train_mse = MSE(y_train, train_pred)
    val_mse = MSE(y_val, val_pred)
    
    return {
        'train_mee':  train_mee,
        'val_mee': val_mee,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'net': net,
        'normalization': (x_min, x_max, y_min, y_max)
    }


def grid_search_cup_kfold(k_folds=5, n_seeds_per_config=1):
    """
    Grid search con k-fold cross validation
    
    Args:
        k_folds: numero di folds per cross validation
        n_seeds_per_config: numero di seed diversi per ogni configurazione
    """
    learning_rates = [0.005, 0.001, 0.002]
    l2_lambdas = [0.00005, 0.0001, 0.0005]
    momentums = [0.9, 0.95]
    architectures = [
        [12, 60, 4],
        [12, 80, 4],
        [12, 50, 30, 4],
        [12, 60, 40, 4],
        [12, 80, 60, 4],
        [12, 60, 40, 20, 4],
    ]

    best_mean_val_mee = float('inf')
    best_config = None
    all_results = []
    
    # Carica dati (useremo train + validation per k-fold)
    X_train_full, y_train_full, X_val_orig, y_val_orig, X_test, y_test = return_CUP(
        dataset_shuffle=True,
        train_size=350,
        validation_size=100,
        test_size=100
    )
    
    # Combina train e validation per k-fold
    X_combined = np.vstack([X_train_full, X_val_orig])
    y_combined = np.vstack([y_train_full, y_val_orig])
    
    total_configs = (len(learning_rates) * len(l2_lambdas) * 
                     len(momentums) * len(architectures))
    total_runs = total_configs * n_seeds_per_config * k_folds
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH CUP CON {k_folds}-FOLD CROSS VALIDATION")
    print(f"{'#'*70}")
    print(f" Configurazioni:  {total_configs}")
    print(f" Seeds per config: {n_seeds_per_config}")
    print(f" Folds per seed: {k_folds}")
    print(f" TOTAL RUNS: {total_runs}")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        for l2 in l2_lambdas:
            for mom in momentums:
                for arch in architectures:
                    print(f"\n{'='*70}")
                    print(f"  CONFIG:  LR={lr}, L2={l2}, Mom={mom}, Arch={arch}")
                    print(f"{'='*70}")
                    
                    config_val_mees = []  # MEE per tutti i seed e folds
                    
                    for seed_idx in range(n_seeds_per_config):
                        seed = seed_idx * 42 + int(lr * 10000) + int(l2 * 100000) + int(mom * 100) + sum(arch)
                        
                        # Crea k-folds con questo seed
                        folds = create_k_folds(X_combined, y_combined, k=k_folds, seed=seed)
                        
                        fold_val_mees = []
                        
                        for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
                            current_run += 1
                            
                            print(f"   Run {current_run}/{total_runs} (Seed {seed_idx+1}/{n_seeds_per_config}, Fold {fold_idx+1}/{k_folds})", end=" → ")
                            
                            try:
                                # Parametri rete - USA MSE per training
                                params = {
                                    'network_structure': arch,
                                    'eta': lr,
                                    'l2_lambda': l2,
                                    'momentum': mom,
                                    'algorithm': 'sgd',
                                    'activation_type': 'tanh',
                                    'loss_type': 'mse',  #  MSE per training
                                    'weight_initializer': 'xavier',
                                    'decay': 0.98,
                                    'mu': 1.75,
                                    'eta_plus':  1.2,
                                    'eta_minus': 0.5,
                                    'debug': False
                                }
                                
                                # Addestra su questo fold
                                fold_results = train_single_fold(
                                    X_train_fold, y_train_fold,
                                    X_val_fold, y_val_fold,
                                    params, verbose=False
                                )
                                
                                fold_val_mees.append(fold_results['val_mee'])
                                config_val_mees.append(fold_results['val_mee'])
                                
                                print(f"Val MEE: {fold_results['val_mee']:.4f}")
                                
                            except Exception as e: 
                                print(f" Error: {str(e)[:40]}")
                                continue
                        
                        # Media su questo seed
                        if len(fold_val_mees) > 0:
                            mean_fold_mee = np.mean(fold_val_mees)
                            std_fold_mee = np.std(fold_val_mees)
                            print(f"     Seed {seed_idx+1} Mean Val MEE: {mean_fold_mee:.4f} ± {std_fold_mee:.4f}")
                    
                    # Media e std su tutti i seed e folds
                    if len(config_val_mees) > 0:
                        mean_val_mee = np.mean(config_val_mees)
                        std_val_mee = np.std(config_val_mees)
                        
                        print(f"\n   CONFIG SUMMARY:  Mean Val MEE = {mean_val_mee:4f} ± {std_val_mee:.4f}")
                        
                        result = {
                            'lr':  lr,
                            'l2_lambda': l2,
                            'momentum': mom,
                            'network_structure': arch,
                            'mean_val_mee': mean_val_mee,
                            'std_val_mee':  std_val_mee,
                            'all_val_mees': config_val_mees
                        }
                        all_results.append(result)
                        
                        # Aggiorna best
                        if mean_val_mee < best_mean_val_mee:
                            best_mean_val_mee = mean_val_mee
                            best_config = result
                            print(f"   NEW BEST:  {best_mean_val_mee:.4f} ± {std_val_mee:.4f}")

    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE (K-FOLD CROSS VALIDATION)")
    print(f"{'='*70}")
    print(f"Mean Validation MEE:    {best_config['mean_val_mee']:.4f} ± {best_config['std_val_mee']:.4f}")
    print(f"Best LR:              {best_config['lr']}")
    print(f"Best L2:              {best_config['l2_lambda']}")
    print(f"Best Momentum:          {best_config['momentum']}")
    print(f"Best Architecture:    {best_config['network_structure']}")
    
    return best_config, all_results, (X_train_full, y_train_full, X_val_orig, y_val_orig, X_test, y_test)


def retrain_final_model(best_config, data_splits, seed=42):
    """
    Re-addestra il modello migliore su train+validation completi
    per la valutazione finale sul test set
    
    Loss function per training:  MSE
    Early stopping basato su:  MEE (validation)
    Tracking: MSE (train) e MEE (validation)
    """
    print(f"\n{'='*70}")
    print(f" RE-TRAINING MODELLO FINALE (TRAIN + VALIDATION)")
    print(f"{'='*70}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits
    
    # Combina train e validation
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    
    # Usa una piccola validation per early stopping (es.  10% dei dati)
    np.random.seed(seed)
    n_val_samples = int(0.1 * X_train_full.shape[0])
    indices = np.arange(X_train_full.shape[0])
    np.random.shuffle(indices)
    
    val_indices = indices[:n_val_samples]
    train_indices = indices[n_val_samples:]
    
    X_train_final = X_train_full[train_indices]
    y_train_final = y_train_full[train_indices]
    X_val_final = X_train_full[val_indices]
    y_val_final = y_train_full[val_indices]
    
    # Normalizzazione
    x_min = X_train_final.min(axis=0)
    x_max = X_train_final.max(axis=0)
    X_train_norm = normalize(X_train_final, 0, 1, x_min, x_max)
    X_val_norm = normalize(X_val_final, 0, 1, x_min, x_max)
    X_test_norm = normalize(X_test, 0, 1, x_min, x_max)
    
    y_min = y_train_final.min(axis=0)
    y_max = y_train_final.max(axis=0)
    y_train_norm = normalize(y_train_final, -1, 1, y_min, y_max)
    y_val_norm = normalize(y_val_final, -1, 1, y_min, y_max)
    y_test_norm = normalize(y_test, -1, 1, y_min, y_max)
    
    
    params = {
        'network_structure': best_config['network_structure'],
        'eta': best_config['lr'],
        'l2_lambda':  best_config['l2_lambda'],
        'momentum': best_config['momentum'],
        'algorithm': 'sgd',
        'activation_type': 'tanh',
        'loss_type': 'mse',  #  MSE per training
        'weight_initializer': 'xavier',
        'decay': 0.98,
        'mu': 1.75,
        'eta_plus':  1.2,
        'eta_minus': 0.5,
        'debug': False
    }
    
    np.random.seed(seed)
    net = NeuralNetwork(**params)
    
    # Training manuale con MSE per training e MEE per early stopping
    train_mse_history = []  #  MSE per training
    val_mee_history = []    #  MEE per validation
    
    epochs = 2000
    batch_size = 16
    patience = 300
    best_val_mee = float('inf')  #  Early stopping su MEE
    patience_counter = 0
    best_weights = None
    
    print(f"\n Training finale in corso (max {epochs} epochs)...")
    print(f" Loss function: MSE (training)")
    print(f" Early stopping: MEE (validation)")
    print(f" Tracking: MSE (train) | MEE (validation)\n")
    
    for epoch in range(epochs):
        try:
            # Training per 1 epoca (la rete ottimizza MSE internamente)
            net.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm,
                   epochs=1, batch_size=batch_size, verbose=False)
        except: 
            if epoch == 0:
                # Fallback se fit(epochs=1) non funziona
                print("  Impossibile fare training epoch-by-epoch, uso fit completo")
                net.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm,
                       epochs=epochs, batch_size=batch_size, patience=patience, verbose=False)
                break
        
        #  Calcola MSE per TRAINING (sui dati ORIGINALI denormalizzati)
        train_pred_norm = net.predict(X_train_norm)
        train_pred = denormalize(train_pred_norm, -1, 1, y_min, y_max)
        train_mse = MSE(y_train_final, train_pred)
        
        #  Calcola MEE per VALIDATION (sui dati ORIGINALI denormalizzati)
        val_pred_norm = net.predict(X_val_norm)
        val_pred = denormalize(val_pred_norm, -1, 1, y_min, y_max)
        val_mee = MEE(y_val_final, val_pred)
        
        train_mse_history.append(train_mse)
        val_mee_history.append(val_mee)
        
        #  Early stopping basato su MEE 
        if val_mee < best_val_mee: 
            best_val_mee = val_mee
            patience_counter = 0
            # Salva i pesi migliori
            try:
                best_weights = [w.copy() for w in net.weights]
            except:
                best_weights = None
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n    Early stopping at epoch {epoch+1}")
            print(f"   Best Val MEE: {best_val_mee:.6f} at epoch {epoch+1-patience}")
            # Ripristina i pesi migliori
            if best_weights is not None:
                try:
                    net.weights = [w.copy() for w in best_weights]
                    print(f"   Restored best weights")
                except:
                    pass
            break
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train MSE: {train_mse:.6f}, Val MEE: {val_mee:.6f}")
    
    print(f"\n Training completato dopo {len(train_mse_history)} epoche")
    print(f"   Final Train MSE: {train_mse_history[-1]:.6f}")
    print(f"   Best Val MEE:     {best_val_mee:.6f}")
    print(f"   Final Val MEE:   {val_mee_history[-1]:.6f}")
    
    # Valutazione test set
    test_pred_norm = net.predict(X_test_norm)
    test_pred = denormalize(test_pred_norm, -1, 1, y_min, y_max)
    
    test_mee = MEE(y_test, test_pred)
    test_mse = MSE(y_test, test_pred)
    
    print(f"\n{'='*70}")
    print(f" RISULTATI FINALI SUL TEST SET")
    print(f"{'='*70}")
    print(f"  Test MEE: {test_mee:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    
    return {
        'net': net,
        'train_mse_history': train_mse_history,  #  MSE
        'val_mee_history': val_mee_history,      #  MEE
        'test_mee':  test_mee,
        'test_mse': test_mse,
        'test_predictions': test_pred,
        'normalization': (x_min, x_max, y_min, y_max),
        'test_data': (X_test, y_test),
        'train_data': (X_train_final, y_train_final),
        'val_data': (X_val_final, y_val_final),
        'best_val_mee': best_val_mee
    }


def plot_kfold_results(best_config, final_results, save_path='cup_kfold_results.png'):
    """
    Crea grafici per i risultati k-fold
    MSE per training, MEE per validation
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: MSE (train) e MEE (validation) vs Epochs
    plt.subplot(1, 3, 1)

    train_mse = final_results['train_mse_history']  #  MSE
    val_mee = final_results['val_mee_history']      #  MEE
    epochs = range(1, len(train_mse) + 1)
    
    train_smooth = smooth_curve(train_mse, weight=0.95)
    val_smooth = smooth_curve(val_mee, weight=0.95)
    
    # Plot con doppio asse Y (MSE e MEE hanno scale diverse)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # MSE su asse sinistro (blu)
    line1 = ax1.plot(epochs, train_smooth, label='Training MSE',
                     color='#1F618D', linewidth=2.5)
    ax1.set_ylabel('MSE (Training)', fontsize=12, fontweight='bold', color='#1F618D')
    ax1.tick_params(axis='y', labelcolor='#1F618D')
    
    # MEE su asse destro (rosso)
    line2 = ax2.plot(epochs, val_smooth, label='Validation MEE',
                     color='#E74C3C', linewidth=2.5)
    ax2.set_ylabel('MEE (Validation)', fontsize=12, fontweight='bold', color='#E74C3C')
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    
    # Optimal epoch basato su MEE
    min_val_idx = np.argmin(val_smooth)
    optimal_epoch = min_val_idx + 1
    ax1.axvline(x=optimal_epoch, color='green', linestyle=': ', linewidth=2.5, alpha=0.7, label=f'Optimal:  {optimal_epoch}')
    ax1.scatter([optimal_epoch], [train_smooth[min_val_idx]],
               color='gold', s=200, marker='*', zorder=10,
               edgecolors='darkgreen', linewidths=2)
    
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training MSE & Validation MEE vs Epochs\n(LR={best_config["lr"]}, L2={best_config["l2_lambda"]})',
                 fontsize=13, fontweight='bold')
    
    # Legenda combinata
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: K-Fold Validation MEE Distribution
    plt.subplot(1, 3, 2)
    
    all_val_mees = best_config['all_val_mees']
    plt.hist(all_val_mees, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(best_config['mean_val_mee'], color='red', linestyle='--', linewidth=2,
               label=f'Mean:  {best_config["mean_val_mee"]:.4f}')
    plt.xlabel('Validation MEE', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'K-Fold CV MEE Distribution\n(μ={best_config["mean_val_mee"]:.4f}, σ={best_config["std_val_mee"]:.4f})',
             fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Test Set Predictions (Target 1)
    plt.subplot(1, 3, 3)
    
    X_test, y_test = final_results['test_data']
    predictions = final_results['test_predictions']
    
    plt.scatter(y_test[: , 0], predictions[:, 0], alpha=0.6, s=50,
               edgecolors='black', linewidth=0.5, color='#2ECC71')
    
    min_val = min(y_test[:, 0].min(), predictions[:, 0].min())
    max_val = max(y_test[:, 0].max(), predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
            linewidth=2, label='Perfect prediction')
    
    plt.xlabel('True Values (Target 1)', fontsize=12, fontweight='bold')
    plt.ylabel('Predictions (Target 1)', fontsize=12, fontweight='bold')
    plt.title(f'Test Set Predictions\nMEE: {final_results["test_mee"]:.4f}',
             fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in:  {save_path}")
    plt.show()


def plot_predictions_vs_actual(best_config, final_results, save_path='cup_predictions_vs_actual.png'):
    """
    Crea un grafico 2x2 con predizioni vs valori reali per tutti i 4 target
    """
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO PREDIZIONI VS VALORI REALI")
    print(f"{'='*70}")
    
    X_test, y_test = final_results['test_data']
    predictions = final_results['test_predictions']
    
    # Grafico 2x2 (4 target)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    target_names = ['Target 1 (x)', 'Target 2 (y)', 'Target 3 (z1)', 'Target 4 (z2)']
    
    for i in range(4):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_test[:, i], predictions[: , i],
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5,
                  color='#3498db', label='Predictions')
        
        # Linea ideale (y=x)
        min_val = min(y_test[:, i].min(), predictions[:, i].min())
        max_val = max(y_test[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2.5, label='Perfect prediction', alpha=0.8)
        
        # Calcola errore per questo target
        target_errors = np.sqrt((y_test[:, i] - predictions[:, i])**2)
        target_mee = np.mean(target_errors)
        
        # Formatting
        ax.set_xlabel(f'True Values - {target_names[i]}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted Values - {target_names[i]}', fontsize=12, fontweight='bold')
        ax.set_title(f'{target_names[i]}\nMean Euclidean Error: {target_mee:.4f}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Imposta limiti uguali per x e y (quadrato)
        all_vals = np.concatenate([y_test[:, i], predictions[:, i]])
        margin = (all_vals.max() - all_vals.min()) * 0.05
        ax.set_xlim(all_vals.min() - margin, all_vals.max() + margin)
        ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)
        ax.set_aspect('equal', 'box')
    
    arch_str = '-'.join(map(str, best_config['network_structure']))
    fig.suptitle(f'Predictions vs True Values - CUP Test Set\n'
                f'(Arch={arch_str}, MEE={final_results["test_mee"]:.4f})',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()


def plot_loss_comparison(final_results, best_config, save_path='cup_loss_comparison.png'):
    """
    Crea un grafico alternativo con MSE e MEE normalizzati sullo stesso asse
    """
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO LOSS COMPARISON (NORMALIZZATO)")
    print(f"{'='*70}")
    
    train_mse = np.array(final_results['train_mse_history'])
    val_mee = np.array(final_results['val_mee_history'])
    epochs = range(1, len(train_mse) + 1)
    
    # Normalizza entrambe le curve su [0, 1] per confrontarle
    train_mse_norm = (train_mse - train_mse.min()) / (train_mse.max() - train_mse.min() + 1e-8)
    val_mee_norm = (val_mee - val_mee.min()) / (val_mee.max() - val_mee.min() + 1e-8)
    
    train_smooth = smooth_curve(list(train_mse_norm), weight=0.95)
    val_smooth = smooth_curve(list(val_mee_norm), weight=0.95)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs, train_smooth, label='Training MSE (normalized)',
           color='#1F618D', linewidth=2.5)
    ax.plot(epochs, val_smooth, label='Validation MEE (normalized)',
           color='#E74C3C', linewidth=2.5)
    
    # Optimal epoch
    min_val_idx = np.argmin(val_smooth)
    optimal_epoch = min_val_idx + 1
    ax.axvline(x=optimal_epoch, color='green', linestyle=':',
              linewidth=2.5, alpha=0.7, label=f'Optimal: {optimal_epoch} epochs')
    ax.scatter([optimal_epoch], [val_smooth[min_val_idx]],
              color='gold', s=400, marker='*', zorder=10,
              edgecolors='darkgreen', linewidths=3)
    
    ax.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Training MSE & Validation MEE (Normalized) vs Epochs\n'
                f'(Arch={best_config["network_structure"]})',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()


def save_results_summary(best_config, final_results, all_results, save_path='cup_results_summary.txt'):
    """
    Salva un riepilogo testuale dei risultati
    """
    print(f"\n{'='*70}")
    print(f" SALVATAGGIO RIEPILOGO RISULTATI")
    print(f"{'='*70}")
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CUP - REGRESSIONE CON K-FOLD CROSS VALIDATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("MIGLIOR CONFIGURAZIONE:\n")
        f.write("-"*70 + "\n")
        f.write(f"Learning Rate:       {best_config['lr']}\n")
        f.write(f"L2 Lambda:          {best_config['l2_lambda']}\n")
        f.write(f"Momentum:           {best_config['momentum']}\n")
        f.write(f"Architecture:       {best_config['network_structure']}\n")
        f.write(f"\nK-Fold CV Mean Val MEE: {best_config['mean_val_mee']:.6f} ± {best_config['std_val_mee']:.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RISULTATI FINALI SUL TEST SET:\n")
        f.write("-"*70 + "\n")
        f.write(f"Test MEE: {final_results['test_mee']:.6f}\n")
        f.write(f"Test MSE: {final_results['test_mse']:.6f}\n")
        f.write(f"Best Val MEE (early stopping): {final_results['best_val_mee']:.6f}\n")
        f.write(f"Final Train MSE: {final_results['train_mse_history'][-1]:.6f}\n")
        f.write(f"Final Val MEE:  {final_results['val_mee_history'][-1]:.6f}\n")
        f.write(f"Training epochs: {len(final_results['train_mse_history'])}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TUTTE LE CONFIGURAZIONI (ordinate per Mean Val MEE):\n")
        f.write("-"*70 + "\n")
        
        # Ordina tutti i risultati per mean_val_mee
        sorted_results = sorted(all_results, key=lambda x: x['mean_val_mee'])
        
        for idx, result in enumerate(sorted_results[: 10]):  # Top 10
            f.write(f"\n{idx+1}.Mean Val MEE: {result['mean_val_mee']:.6f} ± {result['std_val_mee']:.6f}\n")
            f.write(f"   LR={result['lr']}, L2={result['l2_lambda']}, Mom={result['momentum']}\n")
            f.write(f"   Arch={result['network_structure']}\n")
    
    print(f" Riepilogo salvato in: {save_path}")


if __name__ == "__main__": 
    try:
        print("="*70)
        print(" CUP - REGRESSIONE CON K-FOLD CROSS VALIDATION")
        print("="*70)
        
        # FASE 1: Grid search con k-fold CV
        print("\n FASE 1: Grid Search con K-Fold Cross Validation")
        best_config, all_results, data_splits = grid_search_cup_kfold(
            k_folds=5,
            n_seeds_per_config=2  # Usa 1 per velocizzare
        )
        
        # FASE 2: Re-training finale
        print(f"\n FASE 2: Re-training finale su train+validation")
        final_results = retrain_final_model(best_config, data_splits, seed=42)
        
        # FASE 3: Plot risultati principali
        print(f"\n FASE 3: Creazione grafici principali")
        plot_kfold_results(best_config, final_results, save_path='cup_kfold_results.png')
        
        # FASE 4: Plot predizioni vs valori reali (4 target)
        print(f"\n FASE 4: Creazione grafico predizioni vs valori reali")
        plot_predictions_vs_actual(best_config, final_results, save_path='cup_predictions_vs_actual.png')
        
        # FASE 5: Plot loss comparison normalizzato
        print(f"\n FASE 5: Creazione grafico loss comparison")
        plot_loss_comparison(final_results, best_config, save_path='cup_loss_comparison.png')
        
        # FASE 6: Salva riepilogo testuale
        print(f"\n FASE 6: Salvataggio riepilogo risultati")
        save_results_summary(best_config, final_results, all_results, save_path='cup_results_summary.txt')
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print(f"{'='*70}")
        print(f"K-Fold CV Mean Val MEE: {best_config['mean_val_mee']:.4f} ± {best_config['std_val_mee']:.4f}")
        print(f"Test MEE (final):       {final_results['test_mee']:.4f}")
        print(f"Test MSE (final):       {final_results['test_mse']:.4f}")
        print(f"\n Best Parameters:")
        print(f"  LR:            {best_config['lr']}")
        print(f"  L2:           {best_config['l2_lambda']}")
        print(f"  Momentum:     {best_config['momentum']}")
        print(f"  Architecture:  {best_config['network_structure']}")
        
        print(f"\n{'='*70}")
        print(" CUP CON K-FOLD CV COMPLETATO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()