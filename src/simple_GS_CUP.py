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


def _cup_test(learning_rate, seed, l2_lambda=0.001, momentum=0.9, hidden_units=50, verbose=False):
    """
    Esegue un singolo training per CUP (regressione)
    USA SOLO TRAIN E VALIDATION SET (non test!)
    
    Loss per training: MSE (half_mse)
    Metrica per valutazione: MEE (Mean Euclidean Error)
    """
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}, L2: {l2_lambda}, Hidden: {hidden_units}")
    
    # Seed per riproducibilità
    np.random. seed(seed)
    
    # Carica i dati CUP
    X_train, y_train, X_val, y_val, X_test, y_test = return_CUP(
        dataset_shuffle=True,
        train_size=250,
        validation_size=125,
        test_size=125
    )
    
    # Normalizzazione input (0-1)
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    X_train_norm = normalize(X_train, 0, 1, x_min, x_max)
    X_val_norm = normalize(X_val, 0, 1, x_min, x_max)
    X_test_norm = normalize(X_test, 0, 1, x_min, x_max)
    
    # Normalizzazione target (-1, 1) per tanh
    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)
    y_train_norm = normalize(y_train, -1, 1, y_min, y_max)
    y_val_norm = normalize(y_val, -1, 1, y_min, y_max)
    y_test_norm = normalize(y_test, -1, 1, y_min, y_max)
    
    # Configurazione rete
    params = {
        'network_structure': [12, hidden_units, 4],  # 12 input, hidden variabile, 4 output
        'eta': learning_rate,
        'l2_lambda': l2_lambda,
        'momentum': momentum,
        'algorithm': 'sgd',
        'activation_type': 'tanh',  # Tanh per regressione
        'loss_type': 'half_mse',  # MSE per training
        'weight_initializer': 'xavier',
        'decay': 0.95
    }
    
    # Crea e addestra la rete
    net = NeuralNetwork(**params)
    
    history = net.fit(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        epochs=1500,
        batch_size=32,  # Mini-batch per regressione
        patience=150,
        verbose=verbose
    )
    
    # Predizioni
    train_pred_norm = net.predict(X_train_norm)
    val_pred_norm = net.predict(X_val_norm)
    
    # Denormalizza per calcolare MEE
    train_pred = denormalize(train_pred_norm, -1, 1, y_min, y_max)
    val_pred = denormalize(val_pred_norm, -1, 1, y_min, y_max)
    
    # Calcola MEE (metrica di valutazione)
    train_mee = MEE(y_train, train_pred)
    val_mee = MEE(y_val, val_pred)
    
    # Calcola anche MSE per confronto
    train_mse = MSE(y_train, train_pred)
    val_mse = MSE(y_val, val_pred)
    
    # Loss finali (normalizzate)
    if isinstance(history, dict) and 'training' in history: 
        final_train_loss = history['training'][-1] if isinstance(history['training'], list) else history
        final_val_loss = history['validation'][-1] if isinstance(history['validation'], list) else history
    else:
        final_train_loss = history if not isinstance(history, dict) else 0
        final_val_loss = history if not isinstance(history, dict) else 0
    
    return {
        'train_mee': train_mee,
        'val_mee': val_mee,
        'train_mse': train_mse,
        'val_mse':  val_mse,
        'train_loss': final_train_loss,
        'val_loss':  final_val_loss,
        'network': net,
        'history': history,
        'params': params,
        'lr': learning_rate,
        'l2_lambda':  l2_lambda,
        'momentum': momentum,
        'hidden_units': hidden_units,
        'seed': seed,
        'normalization': (x_min, x_max, y_min, y_max),
        'data':  (X_train, y_train, X_val, y_val, X_test, y_test)
    }


def grid_search_cup(n_seeds_per_config=3, learning_rates=None, l2_lambdas=None, 
                    momentums=None, hidden_units_list=None):
    """
    Grid search per CUP (regressione)
    SELEZIONE BASATA SU VALIDATION MEE (non test!)
    """
    if learning_rates is None:
        learning_rates = [0.01, 0.05, 0.1]
    
    if l2_lambdas is None:
        l2_lambdas = [0.0001, 0.001, 0.01]
    
    if momentums is None:
        momentums = [0.85, 0.9, 0.95]
    
    if hidden_units_list is None:
        hidden_units_list = [30, 50, 70]
    
    best_val_mee = float('inf')
    best_results = None
    all_results = []
    
    total_runs = (len(learning_rates) * len(l2_lambdas) * len(momentums) * 
                 len(hidden_units_list) * n_seeds_per_config)
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH CUP:  {len(learning_rates)} LRs × {len(l2_lambdas)} L2s × "
          f"{len(momentums)} Moms × {len(hidden_units_list)} Hidden × {n_seeds_per_config} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        for l2 in l2_lambdas: 
            for mom in momentums:
                for hidden in hidden_units_list:
                    print(f"\n{'='*70}")
                    print(f"🎯 TESTING LR={lr}, L2={l2}, Mom={mom}, Hidden={hidden}")
                    print(f"{'='*70}")
                    
                    for seed_idx in range(n_seeds_per_config):
                        current_run += 1
                        seed = seed_idx * 42 + int(lr * 10000) + int(l2 * 100000) + int(mom * 100) + hidden
                        
                        print(f"\n🔄 Run {current_run}/{total_runs} - LR={lr}, L2={l2}, Mom={mom}, "
                              f"Hidden={hidden}, Seed={seed}", end=" → ")
                        
                        results = _cup_test(learning_rate=lr, seed=seed, l2_lambda=l2, 
                                           momentum=mom, hidden_units=hidden, verbose=False)
                        all_results.append(results)
                        
                        print(f"Train MEE: {results['train_mee']:.4f}, Val MEE: {results['val_mee']:.4f}")
                        
                        if results['val_mee'] < best_val_mee:
                            best_val_mee = results['val_mee']
                            best_results = results
                            print(f"   ✨ NUOVO BEST VAL MEE: {best_val_mee:.4f}")
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE GRID SEARCH")
    print(f"{'='*70}")
    print(f"Validation MEE:   {best_results['val_mee']:.4f}")
    print(f"Train MEE:        {best_results['train_mee']:.4f}")
    print(f"Best LR:         {best_results['lr']}")
    print(f"Best L2:         {best_results['l2_lambda']}")
    print(f"Best Momentum:    {best_results['momentum']}")
    print(f"Best Hidden:     {best_results['hidden_units']}")
    print(f"Best Seed:       {best_results['seed']}")
    
    return best_results, all_results


def retrain_and_track_errors(best_results):
    """
    Ri-addestra il modello migliore tracciando MEE ad ogni epoca
    """
    print(f"\n{'='*70}")
    print(f" RI-ADDESTRAMENTO CON TRACKING DEGLI ERRORI")
    print(f"{'='*70}")
    print(f"LR: {best_results['lr']}, L2: {best_results['l2_lambda']}, "
          f"Hidden: {best_results['hidden_units']}, Seed: {best_results['seed']}")
    
    # Recupera i dati
    X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
    x_min, x_max, y_min, y_max = best_results['normalization']
    
    # Normalizza
    X_train_norm = normalize(X_train, 0, 1, x_min, x_max)
    X_val_norm = normalize(X_val, 0, 1, x_min, x_max)
    y_train_norm = normalize(y_train, -1, 1, y_min, y_max)
    y_val_norm = normalize(y_val, -1, 1, y_min, y_max)
    
    # Seed per riproducibilità
    np.random.seed(best_results['seed'])
    
    # Ricrea la rete
    params = best_results['params']. copy()
    net = NeuralNetwork(**params)
    
    # Training manuale con tracking
    train_mee_history = []
    val_mee_history = []
    
    epochs = 1500
    batch_size = 32
    patience = 150
    best_val_mee = float('inf')
    patience_counter = 0
    
    print(f"\nAddestramento in corso (tracking MEE ad ogni epoca)...")
    
    for epoch in range(epochs):
        try:
            net.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, 
                   epochs=1, batch_size=batch_size, verbose=False)
        except: 
            if epoch == 0:
                net.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, 
                       epochs=epochs, batch_size=batch_size, patience=patience, verbose=False)
            break
        
        # Calcola MEE
        train_pred_norm = net.predict(X_train_norm)
        val_pred_norm = net.predict(X_val_norm)
        
        train_pred = denormalize(train_pred_norm, -1, 1, y_min, y_max)
        val_pred = denormalize(val_pred_norm, -1, 1, y_min, y_max)
        
        train_mee = MEE(y_train, train_pred)
        val_mee = MEE(y_val, val_pred)
        
        train_mee_history. append(train_mee)
        val_mee_history. append(val_mee)
        
        # Early stopping
        if val_mee < best_val_mee:
            best_val_mee = val_mee
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch+1}/{epochs} - Train MEE: {train_mee:.4f}, Val MEE:  {val_mee:.4f}")
    
    print(f"\n Training completato dopo {len(train_mee_history)} epoche")
    print(f"Final Train MEE: {train_mee_history[-1]:.4f}")
    print(f"Final Val MEE:     {val_mee_history[-1]:.4f}")
    
    return train_mee_history, val_mee_history, net


def evaluate_on_test_set(net, X_test, y_test, normalization):
    """
    Valuta il modello finale sul TEST SET (UNA SOLA VOLTA!)
    """
    print(f"\n{'='*70}")
    print(f" VALUTAZIONE FINALE SUL TEST SET")
    print(f"{'='*70}")
    
    x_min, x_max, y_min, y_max = normalization
    
    X_test_norm = normalize(X_test, 0, 1, x_min, x_max)
    test_pred_norm = net.predict(X_test_norm)
    test_pred = denormalize(test_pred_norm, -1, 1, y_min, y_max)
    
    test_mee = MEE(y_test, test_pred)
    test_mse = MSE(y_test, test_pred)
    
    print(f"\n RISULTATI TEST SET:")
    print(f"  Test MEE:  {test_mee:.4f}")
    print(f"  Test MSE:  {test_mse:.4f}")
    
    return {
        'test_mee':  test_mee,
        'test_mse': test_mse,
        'predictions': test_pred
    }


def plot_results(train_results, train_mee_history, val_mee_history, test_results, 
                save_path='cup_best_results.png'):
    """
    Crea grafici completi dei risultati
    """
    train_mee = train_results['train_mee']
    val_mee = train_results['val_mee']
    test_mee = test_results['test_mee']
    lr = train_results['lr']
    l2 = train_results['l2_lambda']
    hidden = train_results['hidden_units']
    seed = train_results['seed']
    
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: MEE vs Epochs
    plt.subplot(1, 3, 1)
    
    if train_mee_history is not None and val_mee_history is not None:
        epochs = range(1, len(train_mee_history) + 1)
        
        smoothing_weight = 0.85
        train_mee_smooth = smooth_curve(train_mee_history, weight=smoothing_weight)
        val_mee_smooth = smooth_curve(val_mee_history, weight=smoothing_weight)
        
        plt.plot(epochs, train_mee_history, linewidth=1, color='#3498db', alpha=0.2, label='_nolegend_')
        plt.plot(epochs, val_mee_history, linewidth=1, color='#e74c3c', alpha=0.2, label='_nolegend_')
        
        plt.plot(epochs, train_mee_smooth, label='Training MEE (smoothed)', 
                linewidth=2.5, color='#3498db')
        plt.plot(epochs, val_mee_smooth, label='Validation MEE (smoothed)', 
                linewidth=2.5, color='#e74c3c')
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('MEE', fontsize=12, fontweight='bold')
        plt.title(f'Training & Validation MEE vs Epochs\n(LR={lr}, L2={l2}, Hidden={hidden}, Seed={seed})', 
                 fontsize=13, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        min_val_mee = np.min(val_mee_history)
        min_val_epoch = np.argmin(val_mee_history) + 1
        plt.scatter([min_val_epoch], [min_val_mee], color='red', s=150, zorder=5, marker='*', 
                   edgecolors='black', linewidths=2, 
                   label=f'Min Val MEE: {min_val_mee:.4f} (epoch {min_val_epoch})')
        
        plt.legend(fontsize=9, loc='best')
    
    # Subplot 2: MEE Bar Chart
    plt.subplot(1, 3, 2)
    
    categories = ['Train', 'Validation', 'Test']
    mees = [train_mee, val_mee, test_mee]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = plt.bar(categories, mees, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    for i, (cat, mee) in enumerate(zip(categories, mees)):
        plt.text(i, mee + 0.05, f'{mee:.4f}', ha='center', fontsize=13, fontweight='bold')
    
    plt.ylabel('MEE', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance (CUP Regression)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Subplot 3: Predictions scatter (solo un target per visualizzazione)
    plt.subplot(1, 3, 3)
    
    X_train, y_train, X_val, y_val, X_test, y_test = train_results['data']
    predictions = test_results['predictions']
    
    # Plot solo il primo target
    plt.scatter(y_test[: , 0], predictions[:, 0], alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Linea ideale (y=x)
    min_val = min(y_test[:, 0]. min(), predictions[:, 0].min())
    max_val = max(y_test[:, 0].max(), predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    plt.xlabel('True Values (Target 1)', fontsize=12, fontweight='bold')
    plt.ylabel('Predictions (Target 1)', fontsize=12, fontweight='bold')
    plt.title(f'Test Set Predictions vs True Values\nMEE: {test_mee:.4f}', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()

def plot_bias_variance_tradeoff(all_results, complexity_param='hidden_units', 
                                dataset_name='MONK-1',
                                save_path='bias_variance_tradeoff. png'):
    """
    Crea il grafico Bias-Variance Tradeoff classico
    Mostra training e test error vs complessità del modello
    
    Args:
        all_results: lista di risultati da grid search
        complexity_param: 'hidden_units', 'epochs', o 'lr'
        dataset_name: nome del dataset per il titolo
        save_path: path del file output
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO BIAS-VARIANCE TRADEOFF")
    print(f"{'='*70}")
    print(f"Complexity parameter: {complexity_param}")
    print(f"Numero totale run: {len(all_results)}")
    
    # ========================================
    # RAGGRUPPA RISULTATI PER COMPLESSITÀ
    # ========================================
    complexity_groups = defaultdict(lambda: {'train': [], 'test': [], 'val': []})
    
    for result in all_results:
        # Determina la complessità
        if complexity_param == 'hidden_units': 
            complexity = result. get('hidden_units', result['params']['network_structure'][1])
        elif complexity_param == 'epochs':
            if 'history' in result and isinstance(result['history'], dict):
                complexity = len(result['history']. get('training', [100]))
            else:
                complexity = 100
        elif complexity_param == 'lr': 
            complexity = result['lr']
        else:
            complexity = result. get(complexity_param, 0)
        
        # Determina training e test error
        if 'train_mee' in result:  # CUP (regressione)
            train_error = result['train_mee']
            val_error = result.get('val_mee', None)
            test_error = result.get('test_mee', None)
        else:  # MONK (classificazione)
            train_error = 1 - result['train_accuracy']
            val_error = 1 - result.get('val_accuracy', 0)
            test_error = 1 - result.get('test_accuracy', val_error)
        
        complexity_groups[complexity]['train'].append(train_error)
        if val_error is not None: 
            complexity_groups[complexity]['val'].append(val_error)
        if test_error is not None:
            complexity_groups[complexity]['test'].append(test_error)
    
    # Ordina per complessità
    complexities = sorted(complexity_groups.keys())
    
    print(f"Complessità trovate: {complexities}")
    print(f"Run per complessità: {[len(complexity_groups[c]['train']) for c in complexities]}")
    
    # ========================================
    # PLOT
    # ========================================
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # ========================================
    # PLOT LINEE SOTTILI (ogni singolo run)
    # ========================================
    max_runs = max(len(complexity_groups[c]['train']) for c in complexities)
    
    print(f"\nPlotting {max_runs} run individuali...")
    
    for run_idx in range(max_runs):
        train_path = []
        test_path = []
        x_path = []
        
        for complexity in complexities:
            if run_idx < len(complexity_groups[complexity]['train']):
                x_path.append(complexity)
                train_path. append(complexity_groups[complexity]['train'][run_idx])
                
                if complexity_groups[complexity]['test'] and \
                   run_idx < len(complexity_groups[complexity]['test']):
                    test_path.append(complexity_groups[complexity]['test'][run_idx])
        
        # Linee sottili blu (training)
        if len(x_path) > 1:
            ax.plot(x_path, train_path, color='lightblue', alpha=0.25, 
                   linewidth=0.7, zorder=1)
        
        # Linee sottili rosse (test)
        if len(test_path) > 1 and len(test_path) == len(x_path):
            ax.plot(x_path, test_path, color='lightcoral', alpha=0.25, 
                   linewidth=0.7, zorder=1)
    
    # ========================================
    # PLOT LINEE SPESSE (medie)
    # ========================================
    train_means = []
    test_means = []
    train_stds = []
    test_stds = []
    
    for complexity in complexities: 
        train_means.append(np.mean(complexity_groups[complexity]['train']))
        train_stds.append(np.std(complexity_groups[complexity]['train']))
        
        if complexity_groups[complexity]['test']:
            test_means.append(np.mean(complexity_groups[complexity]['test']))
            test_stds. append(np.std(complexity_groups[complexity]['test']))
        else:
            test_means. append(np.nan)
            test_stds.append(np.nan)
    
    # Linea spessa BLU (training mean)
    ax.plot(complexities, train_means, color='#1F618D', linewidth=4, 
           label='Training Error (mean)', zorder=10)
    
    # Linea spessa ROSSA (test mean)
    test_clean_indices = [i for i, m in enumerate(test_means) if not np.isnan(m)]
    if test_clean_indices:
        test_c = [complexities[i] for i in test_clean_indices]
        test_m = [test_means[i] for i in test_clean_indices]
        ax.plot(test_c, test_m, color='#CB4335', linewidth=4, 
               label='Test Error (mean)', zorder=10)
    
    # ========================================
    # ANNOTAZIONI
    # ========================================
    all_errors = train_means + [m for m in test_means if not np.isnan(m)]
    y_max = max(all_errors) if all_errors else 1. 0
    y_min = min(all_errors) if all_errors else 0.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    # High Bias, Low Variance (sinistra)
    text_x_left = min(complexities) + (max(complexities) - min(complexities)) * 0.08
    ax.text(text_x_left, y_max - 0.03 * y_range,
           'High Bias\nLow Variance', fontsize=11, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8,
                    edgecolor='orange', linewidth=1.5))
    
    # Low Bias, High Variance (destra)
    text_x_right = max(complexities) - (max(complexities) - min(complexities)) * 0.08
    ax.text(text_x_right, y_max - 0.03 * y_range,
           'Low Bias\nHigh Variance', fontsize=11, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8,
                    edgecolor='red', linewidth=1.5))
    
    # Lucky/Unlucky
    ax.text(max(complexities) * 0.97, y_min + 0.15 * y_range, 
           'lucky', fontsize=11, style='italic', color='#1F618D', 
           ha='right', fontweight='bold')
    ax.text(max(complexities) * 0.97, y_max - 0.25 * y_range, 
           'unlucky', fontsize=11, style='italic', color='#CB4335', 
           ha='right', fontweight='bold')
    
    # Optimal complexity (minimo test error medio)
    if test_clean_indices:
        test_means_clean = [test_means[i] for i in test_clean_indices]
        min_test_idx = np.argmin(test_means_clean)
        optimal_complexity = test_c[min_test_idx]
        optimal_error = test_m[min_test_idx]
        
        ax.axvline(x=optimal_complexity, color='green', linestyle=':', 
                  linewidth=2.5, alpha=0.7, zorder=9,
                  label=f'Optimal:  {optimal_complexity}')
        
        ax.scatter([optimal_complexity], [optimal_error], color='green', 
                  s=200, marker='*', zorder=11, edgecolors='black', linewidths=2)
    
    # ========================================
    # FORMATTING
    # ========================================
    xlabel_map = {
        'hidden_units': 'Model Complexity (Hidden Units)',
        'epochs': 'Model Complexity (Training Epochs)',
        'lr': 'Learning Rate'
    }
    
    ax.set_xlabel(xlabel_map.get(complexity_param, f'Model Complexity ({complexity_param})'), 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Prediction Error', fontsize=13, fontweight='bold')
    ax.set_title(f'Bias-Variance Tradeoff - {dataset_name}\nTraining and Test Error vs Model Complexity', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Limiti assi
    ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.15 * y_range)
    ax.set_xlim(min(complexities) - (max(complexities) - min(complexities)) * 0.02,
                max(complexities) + (max(complexities) - min(complexities)) * 0.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Bias-Variance salvato in: {save_path}")
    plt.show()


if __name__ == "__main__": 
    try:
        print("="*70)
        print(" CUP - REGRESSIONE CON GRID SEARCH")
        print("="*70)
        
        # FASE 1: Grid search
        print("\n FASE 1: Grid Search (Train + Validation)")
        best_results, all_results = grid_search_cup(
            n_seeds_per_config=2,
            learning_rates=[0.01, 0.05, 0.1],
            l2_lambdas=[0.0001, 0.001, 0.01],
            momentums=[0.85, 0.9, 0.95],
            hidden_units_list=[30, 50, 70]
        )
        
        # FASE 2: Re-training con tracking
        print(f"\n FASE 2: Re-training con Error Tracking")
        train_mee_history, val_mee_history, final_net = retrain_and_track_errors(best_results)
        
        # FASE 3: Valuta sul TEST SET
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        test_results = evaluate_on_test_set(final_net, X_test, y_test, best_results['normalization'])
        
        # FASE 4: Plot
        plot_results(best_results, train_mee_history, val_mee_history, test_results, 
                    save_path='cup_best_results.png')
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        print(f"\n{'─'*70}")
        print(" MEE SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train MEE:          {best_results['train_mee']:.4f}")
        print(f"  Validation MEE:    {best_results['val_mee']:.4f}")
        print(f"  Test MEE (FINAL): {test_results['test_mee']:.4f}")
        
        print(f"\n{'─'*70}")
        print(" MSE:")
        print(f"{'─'*70}")
        print(f"  Train MSE:  {best_results['train_mse']:.4f}")
        print(f"  Val MSE:    {best_results['val_mse']:.4f}")
        print(f"  Test MSE:    {test_results['test_mse']:.4f}")
        
        # PARAMETRI COMPLETI
        params = best_results['params']
        print(f"\n{'='*70}")
        print(" PARAMETRI DELLA RETE NEURALE")
        print(f"{'='*70}")
        
        print(f"\n ARCHITETTURA:")
        print(f"  Network Structure:     {params['network_structure']}")
        print(f"    - Input Layer:       {params['network_structure'][0]} neurons")
        print(f"    - Hidden Layer:     {params['network_structure'][1]} neurons")
        print(f"    - Output Layer:       {params['network_structure'][-1]} neurons")
        
        print(f"\n TRAINING:")
        print(f"  Algorithm:           {params['algorithm']. upper()}")
        print(f"  Learning Rate:       {params['eta']}")
        print(f"  Momentum:            {params['momentum']}")
        print(f"  Batch Size:          32")
        print(f"  Max Epochs:          1500")
        print(f"  Actual Epochs:       {len(train_mee_history)}")
        
        print(f"\n REGOLARIZZAZIONE:")
        print(f"  L2 Lambda:            {params['l2_lambda']}")
        
        print(f"\n FUNZIONI:")
        print(f"  Activation:           {params['activation_type']}")
        print(f"  Loss (training):     {params['loss_type']} (MSE)")
        print(f"  Metric (eval):       MEE (Mean Euclidean Error)")
        
        print(f"\n RIPRODUCIBILITÀ:")
        print(f"  Random Seed:         {best_results['seed']}")
        
        print(f"\n DATASET:")
        print(f"  Training Set:        250 esempi")
        print(f"  Validation Set:      125 esempi")
        print(f"  Test Set:            125 esempi")
        print(f"  Input Features:      12")
        print(f"  Output Targets:      4 (t1, t2, t3, t4)")
        
        # PARAMETRI FINALI RETE
        if hasattr(final_net, 'layers'):
            print(f"\n{'='*70}")
            print(" STATISTICHE PESI FINALI")
            print(f"{'='*70}")
            
            total_weights = 0
            for l_idx in range(1, len(final_net.layers)):
                layer_weights = []
                layer_biases = []
                for neuron in final_net.layers[l_idx]:
                    if hasattr(neuron, 'weights'):
                        layer_weights.extend(neuron.weights)
                        layer_biases.append(neuron.bias)
                
                if layer_weights:
                    layer_weights = np.array(layer_weights)
                    layer_biases = np.array(layer_biases)
                    total_weights += len(layer_weights) + len(layer_biases)
                    
                    print(f"\n  Layer {l_idx}:")
                    print(f"    Peso medio:    {np.mean(layer_weights):.6f}")
                    print(f"    Peso std:     {np.std(layer_weights):.6f}")
                    print(f"    Peso min/max: {np.min(layer_weights):.6f} / {np.max(layer_weights):.6f}")
            
            print(f"\n  Totale parametri:  {total_weights}")
        
        print(f"\n{'='*70}")
        print(" CONCLUSIONI")
        print(f"{'='*70}")
        
        if test_results['test_mee'] < 2.0:
            print(" Risultato ECCELLENTE! Test MEE < 2.0")
        elif test_results['test_mee'] < 3.0:
            print(" Risultato BUONO! Test MEE < 3.0")
        else:
            print(f" Test MEE: {test_results['test_mee']:.4f}")
        
        print(f"\n File salvati:")
        print(f"  - cup_best_results.png (grafici)")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO CUP COMPLETATO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback. print_exc()