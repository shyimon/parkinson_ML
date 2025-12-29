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


def _cup_test(learning_rate, seed, l2_lambda=0.0001, momentum=0.9, hidden_units=50, verbose=False):
    
    if verbose:
        print(f" LR: {learning_rate}, L2: {l2_lambda}, Momentum: {momentum}, Hidden: {hidden_units}")
    
    np.random.seed(seed)
    
    X_train, y_train, X_val, y_val, X_test, y_test = return_CUP(
        dataset_shuffle=True,
        train_size=350,
        validation_size=100,
        test_size=100
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
    
    params = {
        'network_structure': [12, hidden_units, 4], 
        'eta': learning_rate,
        'l2_lambda': l2_lambda,
        'momentum': momentum,
        'algorithm': 'sgd',
        'activation_type': 'tanh',  # Tanh per regressione
        'loss_type': 'half_mse',  # MSE per training
        'weight_initializer': 'xavier',
        'decay': 0.98,
        'mu': 1.75,
        'eta_plus': 1.2,
        'eta_minus': 0.5,
        'debug': False

    }
    
    net = NeuralNetwork(**params)
    
    history = net.fit(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        epochs=2000,
        batch_size=16,  # Mini-batch per regressione
        patience=300,
        verbose=verbose
    )
    train_loss_history = []
    val_loss_history = []

    if isinstance(history, dict):
        if 'training' in history:
            if isinstance(history['training'], list):
                train_loss_history = history['training']
            elif isinstance(history['training'], (int, float)):
                tain_loss_history = [history['training']]
        if 'validation' in history:
            if isinstance(history['validation'], list):
                val_loss_history = history['validation']
            elif isinstance(history['validation'], (int, float)):
                val_loss_history = [history['validation']]
    elif isinstance(history, (list, tuple)):
        if len(history) == 2:
            train_loss_history = history[0] if isinstance(history[0], list) else [history[0]]
            val_loss_history = history[1] if isinstance(history[1], list) else [history[1]]
    if len(train_loss_history) == 0:
        print(" History vuota, usando loss finale")
        train_pred_norm = net.predict(X_train_norm)
        train_loss = 0.5 * np.mean((train_pred_norm - y_train_norm) ** 2)
        train_loss_history = [train_loss]
    
        val_pred_norm = net.predict(X_val_norm)
        val_loss = 0.5 * np.mean((val_pred_norm - y_val_norm) ** 2)
        val_loss_history = [val_loss] 

    # Predizioni
    train_pred_norm = net.predict(X_train_norm)
    val_pred_norm = net.predict(X_val_norm)
    
    # Denormalizza per calcolare MEE
    train_pred = denormalize(train_pred_norm, -1, 1, y_min, y_max)
    val_pred = denormalize(val_pred_norm, -1, 1, y_min, y_max)
    
    
    train_mee = MEE(y_train, train_pred)
    val_mee = MEE(y_val, val_pred)
    train_mse = MSE(y_train, train_pred)
    val_mse = MSE(y_val, val_pred)
    
    # loss history
    if isinstance(history, dict) and 'training' in history: 
        train_loss_history = history['training'] if isinstance(history['training'], list) else [history['training']]
        val_loss_history = history['validation'] if isinstance(history['validation'], list) else [history['validation']]
    else:
        train_loss_history = []
        val_loss_history = []
    
    return {
        'train_mee': train_mee,
        'val_mee': val_mee,
        'train_mse': train_mse,
        'val_mse':  val_mse,
        'train_loss_history': train_loss_history,  
        'val_loss_history':  val_loss_history,  
        'network':  net,
        'history': history,
        'params':  params,
        'lr': learning_rate,
        'l2_lambda': l2_lambda,
        'momentum': momentum,
        'hidden_units': hidden_units,
        'seed': seed,
        'normalization': (x_min, x_max, y_min, y_max),
        'data':  (X_train, y_train, X_val, y_val, X_test, y_test)
    }

def grid_search_cup(n_seeds_per_config=3):
    learning_rates = [0.005, 0.001, 0.002]
    l2_lambdas = [0.00005, 0.0001, 0.0005]
    momentums = [0.9, 0.95]
    hidden_units_list=[40, 60, 80]

    best_val_mee = float('inf')
    best_results = None
    all_results = []
    
    total_runs = (len(learning_rates) * len(l2_lambdas) * len(momentums) * 
                 len(hidden_units_list) * n_seeds_per_config)
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH CUP: {total_runs} CONFIGURAZIONI TOTALI")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        for l2 in l2_lambdas: 
            for mom in momentums:
                for hidden in hidden_units_list:
                    print(f"\n{'='*70}")
                    print(f" TESTING LR={lr}, L2={l2}, Mom={mom}, Hidden={hidden}")
                    print(f"{'='*70}")
                    
                    for seed_idx in range(n_seeds_per_config):
                        current_run += 1
                        seed = seed_idx * 42 + int(lr * 10000) + int(l2 * 100000) + int(mom * 100) + hidden
                        
                        print(f"\n Run {current_run}/{total_runs} - LR={lr}, L2={l2}, Mom={mom}, "
                              f"Hidden={hidden}, Seed={seed}", end=" → ")
                        try:
                            results = _cup_test(learning_rate=lr, seed=seed, l2_lambda=l2, 
                                               momentum=mom, hidden_units=hidden, verbose=False)
                            all_results.append(results)
                            print(f"Val MEE: {results['val_mee']:.4f}")
                            
                            if results['val_mee'] < best_val_mee:
                                best_val_mee = results['val_mee']
                                best_results = results
                                print(f"   NUOVO BEST VAL MEE: {best_val_mee:.4f}")
                        except Exception as e:
                            print(f"Error: {str(e)[:40]}")
                            continue

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
    params = best_results['params'].copy()
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
        
        train_mee_history.append(train_mee)
        val_mee_history.append(val_mee)
        
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
    min_val = min(y_test[:, 0].min(), predictions[:, 0].min())
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

def plot_loss_vs_epochs(best_results, save_path='cup_loss_vs_epochs.png'):
    
    print(f"\n{'='*70}")
    print(f"📊 CREAZIONE GRAFICO LOSS VS EPOCHS")
    print(f"{'='*70}")
    
    train_loss_history = best_results. get('train_loss_history', [])
    val_loss_history = best_results.get('val_loss_history', [])
    
    if len(train_loss_history) == 0 or len(val_loss_history) == 0:
        print("⚠️  Nessuna history disponibile!")
        return
    
    epochs = range(1, len(train_loss_history) + 1)
    
    # SMOOTHING
    def smooth_curve(values, weight=0.98):
        smoothed = []
        last = values[0] if values else 0
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    train_smooth = smooth_curve(train_loss_history, weight=0.98)
    val_smooth = smooth_curve(val_loss_history, weight=0.98)
    
    # PLOT
    fig, ax = plt.subplots(figsize=(11, 7))
    
    ax.plot(epochs, train_smooth, label='Training Loss', 
           color='#1F618D', linewidth=2)
    
    ax.plot(epochs, val_smooth, label='Validation Loss', 
           color='#E74C3C', linewidth=2)
    
    # OPTIMAL EPOCH (minimo validation loss)
    min_val_idx = np.argmin(val_smooth)
    optimal_epoch = min_val_idx + 1
    optimal_loss = val_smooth[min_val_idx]
    
    ax.axvline(x=optimal_epoch, color='green', linestyle=':', 
              linewidth=2.5, alpha=0.7, label=f'Optimal:  {optimal_epoch} epochs')
    
    ax.scatter([optimal_epoch], [optimal_loss], color='gold', 
              s=400, marker='*', zorder=11, edgecolors='darkgreen', linewidths=3)
    
    # FORMATTING
    ax.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Half MSE Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Training and Validation Loss vs Epochs - CUP\n'
                f'(LR={best_results["lr"]}, L2={best_results["l2_lambda"]}, '
                f'Hidden={best_results["hidden_units"]})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # LIMITI
    y_max = max(max(train_smooth), max(val_smooth))
    y_min = min(min(train_smooth), min(val_smooth))
    y_range = y_max - y_min if y_max > y_min else 0.001
    
    ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.15 * y_range)
    ax.set_xlim(0, len(epochs) * 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()
    
if __name__ == "__main__": 
    try:
        print("="*70)
        print(" CUP - REGRESSIONE")
        print("="*70)
        
        # FASE 1: Grid search
        print("\n FASE 1: Grid Search")
        best_results, all_results = grid_search_cup(n_seeds_per_config=3)
        
        # ← AGGIUNGI QUESTO DEBUG: 
        print(f"\n DEBUG best_results keys: {best_results.keys()}")
        print(f" train_loss_history length: {len(best_results. get('train_loss_history', []))}")
        print(f" val_loss_history length: {len(best_results.get('val_loss_history', []))}")

        if len(best_results. get('train_loss_history', [])) > 0:
            print(f" train_loss_history primi 5: {best_results['train_loss_history'][:5]}")
        
        # FASE 2: Grafico Loss vs Epochs
        print(f"\n FASE 2: Grafico Loss vs Epochs")
        plot_loss_vs_epochs(best_results, save_path='cup_loss_vs_epochs.png')
        
        # FASE 3: Test Set
        print(f"\n FASE 3: Valutazione Test Set")
        X_test = best_results['data'][4]
        y_test = best_results['data'][5]
        test_results = evaluate_on_test_set(best_results['network'], X_test, y_test, 
                                            best_results['normalization'])
        
        # RIEPILOGO
        print(f"\n{'='*70}")
        print(" RIEPILOGO")
        print(f"{'='*70}")
        print(f"Train MEE:   {best_results['train_mee']:.4f}")
        print(f"Val MEE:     {best_results['val_mee']:.4f}")
        print(f"Test MEE:   {test_results['test_mee']:.4f}")
        print(f"\nParametri:")
        print(f"  LR:      {best_results['lr']}")
        print(f"  L2:       {best_results['l2_lambda']}")
        print(f"  Momentum: {best_results['momentum']}")
        print(f"  Hidden:   {best_results['hidden_units']}")
        
        print(f"\n{'='*70}")
        print(" CUP COMPLETATO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()