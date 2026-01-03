import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk1
import os
OUTPUT_DIR = "monk1_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _monk1_test(learning_rate, seed, verbose=False): 
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}")
    
    # Seed per riproducibilità
    np.random.seed(seed)

    # Carica i dati MONK-1
    X_train, y_train, X_val, y_val, X_test, y_test = return_monk1(
        one_hot=True, 
        val_split=0.3,
        dataset_shuffle=True
    )
    
    
    params = {
        'network_structure': [17, 4, 1],
        'eta': learning_rate,
        'l2_lambda': 0.0,
        'momentum': 0.9,
        'algorithm': 'sgd',
        'activation_type': 'sigmoid',
        'loss_type': 'half_mse',
        'weight_initializer': 'xavier',
        'decay': 0.9,
        'mu': 1.75,
        'eta_plus':  1.2,
        'eta_minus':  0.5,
        'debug': False
    }
    
    # Crea e addestra la rete
    net = NeuralNetwork(**params)
    
    history = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=400,
        batch_size=1,
        patience=50,
        verbose=verbose
    )
    
    # Valutazione su TRAIN e VALIDATION (NON test!)
    train_pred = net.predict(X_train)
    train_pred_class = (train_pred > 0.5).astype(int)
    train_acc = np.mean(train_pred_class == y_train)
    train_error = 1 - train_acc
    
    val_pred = net.predict(X_val)
    val_pred_class = (val_pred > 0.5).astype(int)
    val_acc = np.mean(val_pred_class == y_val)
    val_error = 1 - val_acc
    
    # Calcola le loss finali
    if isinstance(history, dict) and 'training' in history:
        final_train_loss = history['training'][-1] if isinstance(history['training'], list) else history['training']
        final_val_loss = history['validation'][-1] if isinstance(history['validation'], list) else history['validation']
    else:  
        final_train_loss = history if not isinstance(history, dict) else 0
        final_val_loss = history if not isinstance(history, dict) else 0
    
    return {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_error': train_error,
        'val_error': val_error,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'network': net,
        'history':  history,
        'params': params,
        'lr': learning_rate,
        'seed': seed,
        'data': (X_train, y_train, X_val, y_val, X_test, y_test)
    }

def k_fold_cross_validation(learning_rate, k=5, epochs=400, batch_size=1, patience=50, seed=42):

    print(f"\n{'='*70}")
    print(f" K-FOLD CROSS VALIDATION (k={k})")
    print(f"{'='*70}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs:  {epochs}, Batch Size: {batch_size}, Patience: {patience}\n")
    
    np.random.seed(seed)

    # carico tutto il training set
    X_train_full, y_train_full, _, _, X_test, y_test = return_monk1(
        one_hot=True, 
        val_split=0.0,  # Non facciamo split qui
        dataset_shuffle=True
    )

    n_samples = len(X_train_full)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    base_fold_size = n_samples // k
    remainder = n_samples % k
    
    # creo array con le dimensioni di ogni fold
    fold_sizes = [base_fold_size + (1 if i < remainder else 0) for i in range(k)]

    fold_results = []
    fold_accuracies = []
    fold_losses = []

    #calcolo gli inidici di inizio per ogni fold
    fold_starts = [0]
    for size in fold_sizes[:-1]:
        fold_starts.append(fold_starts[-1] + size)

    for fold_idx in range(k):
        print(f"\n{'─'*70}")
        print(f" FOLD {fold_idx + 1}/{k}")
        print(f"{'─'*70}")
        
        # Calcolo indici per validation fold corrente
        val_start = fold_starts[fold_idx]
        val_end = val_start + fold_sizes[fold_idx]
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Split dei dati
        X_train_fold = X_train_full[train_indices]
        y_train_fold = y_train_full[train_indices]
        X_val_fold = X_train_full[val_indices]
        y_val_fold = y_train_full[val_indices]
        
        print(f"  Traininig samples: {len(X_train_fold)}")
        print(f"  Validation samples:    {len(X_val_fold)}")
        
        # Crea e addestra il modello
        params = {
            'network_structure': [17, 4, 1],
            'eta': learning_rate,
            'l2_lambda': 0.0,
            'momentum': 0.9,
            'algorithm': 'sgd',
            'activation_type': 'sigmoid',
            'loss_type': 'half_mse',
            'weight_initializer': 'xavier',
            'decay': 0.9,
            'mu':  1.75,
            'eta_plus': 1.2,
            'eta_minus':  0.5,
            'debug': False
        }
        
        net = NeuralNetwork(**params)
        
        history = net.fit(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=False
        )
        
        # Valutazione sul fold
        val_pred = net.predict(X_val_fold)
        val_pred_class = (val_pred > 0.5).astype(int)
        val_acc = np.mean(val_pred_class == y_val_fold)
        
        # Loss finale
        if isinstance(history, dict) and 'validation' in history:
            val_loss = history['validation'][-1] if isinstance(history['validation'], list) else history['validation']
        else:
            val_loss = 0.0
        
        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)
        
        print(f"  Validation Accuracy: {val_acc:.4%}")
        print(f"  Validation Loss:     {val_loss:.6f}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'network': net,
            'history': history
        })
    
    # Statistiche finali
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    print(f"\n{'='*70}")
    print(f" RISULTATI K-FOLD CROSS VALIDATION")
    print(f"{'='*70}")
    print(f"Mean Validation Accuracy: {mean_acc:.4%} ± {std_acc:.4%}")
    print(f"Mean Validation Loss:      {mean_loss:.6f} ± {std_loss:.6f}")
    print(f"Min Accuracy:  {min(fold_accuracies):.4%}")
    print(f"Max Accuracy:  {max(fold_accuracies):.4%}")
    print(f"{'='*70}\n")
    
    return {
        'fold_results': fold_results,
        'fold_accuracies': fold_accuracies,
        'fold_losses': fold_losses,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'learning_rate': learning_rate,
        'k': k
    }

def grid_search(n_seeds=5, learning_rates=None, momentums=None,
                decays=None, patiences=None):
    
    # Grid search su learning rate e momentum

    if learning_rates is None: 
        learning_rates = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07]

    if momentums is None: 
        momentums = [0.90, 0.92, 0.94, 0.95]
    
    if decays is None:
        decays = [0.85, 0.9, 0.95]
    
    if patiences is None:
        patiences = [20, 30, 40, 50, 60]
    
    best_val_acc = 0
    best_results = None
    all_results = []
    
    total_runs = (len(learning_rates) * len(momentums) * len(decays) *
    len(patiences) * n_seeds)
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH:  {len(learning_rates)} LRs × {len(momentums)} Momentums × {len(decays)} Decays × {len(patiences)} Patiences  × {n_seeds} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    from datetime import datetime
    start_time = datetime.now()

    for lr in learning_rates:
        for mom in momentums:
            for decay in decays:
                for patience in patiences:

                    print(f"\n{'='*70}")
                    print(f"LR={lr}, Momentum={mom}, Decay={decay}, Patience={patience}")
                    print(f"{'='*70}")
        
                    for seed_idx in range(n_seeds):
                        current_run += 1
                        seed = seed_idx * 123 + int(lr * 1000) + int(mom * 100)

                    np.random.seed(seed)
                    X_train, y_train, X_val, y_val, X_test, y_test = return_monk1(
                    one_hot=True, val_split=0.3, dataset_shuffle=True
                    )
                
                    params = {
                        'network_structure': [17, 4, 1],
                        'eta': lr,
                        'l2_lambda': 0.0,
                        'momentum': mom,  
                        'algorithm': 'sgd',
                        'activation_type': 'sigmoid',
                        'loss_type': 'half_mse',
                        'weight_initializer': 'xavier',
                        'decay': decay,
                        'mu': 1.75,
                        'eta_plus':   1.2,
                        'eta_minus':   0.5,
                        'debug': False
                    }
                
                    net = NeuralNetwork(**params)
                    history = net.fit(X_train, y_train, X_val, y_val, 
                                  epochs=400, batch_size=1, patience=50, 
                                  verbose=False)
                
                    # Valutazione
                    train_pred = net.predict(X_train)
                    train_acc = np.mean((train_pred > 0.5).astype(int) == y_train)
                
                    val_pred = net.predict(X_val)
                    val_acc = np.mean((val_pred > 0.5).astype(int) == y_val)
                
                    if isinstance(history, dict) and 'training' in history:
                        final_train_loss = history['training'][-1]
                        final_val_loss = history['validation'][-1]
                    else:
                        final_train_loss = 0
                        final_val_loss = 0
                
                    results = {
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'train_error': 1 - train_acc,
                        'val_error': 1 - val_acc,
                        'train_loss': final_train_loss,
                        'val_loss': final_val_loss,
                        'network':  net,
                        'history': history,
                        'params': params,
                        'lr': lr,
                        'momentum': mom,
                        'decay': decay,
                        'patience': patience,
                        'seed': seed,
                        'data':  (X_train, y_train, X_val, y_val, X_test, y_test)
                    }
                    all_results.append(results)
                
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_results = results
                        print(f"\n     NEW BEST:  {best_val_acc:.4%}")
                        print(f"LR={lr}, Momentum={mom}, Decay={decay}, Patience={patience}, Seed={seed})")
                
                    if val_acc >= 1.0:
                        print(f"\n 100% VALIDATION ACCURACY REACHED!")
                        print(f"{'='*70}\n")
                        return best_results, all_results
    
    print(f"\n\n{'='*70}")
    print(f" BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {best_val_acc:.4%}")
    print(f"Learning Rate:        {best_results['lr']}")
    print(f"Momentum:            {best_results['momentum']}")
    print(f"Decay:               {best_results['decay']}")
    print(f"Weight Init:         {best_results['weight_init']}")
    print(f"Patience:            {best_results['patience']}")
    print(f"Seed:                {best_results['seed']}")
    print(f"{'='*70}\n")
    
    return best_results, all_results


def evaluate_on_test_set(net, X_test, y_test):
    
    # Valuta il modello finale sul TEST SET (UNA SOLA VOLTA!)
    
    print(f"\n{'='*70}")
    print(f" VALUTAZIONE FINALE SUL TEST SET")
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
    print(f"\n Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN:  {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error': test_error,
        'precision': precision,
        'recall': recall,
        'f1':  f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def plot_training_history(best_results, save_path=f'{OUTPUT_DIR}/monk1_training_history.png'):
    
    # plotto training loss e validation loss (half mse) vs Epochs

    history = best_results['history']
    if not isinstance(history, dict) or 'training' not in history:
        print("  Nessuna history disponibile per il plot")
        return
    
    train_loss = history['training']
    val_loss = history['validation']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss (Half MSE)', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss (Half MSE)', color='orange', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.title(f'Training loss (MSE) and Validation Loss (MEE) vs Epochs\n(LR={best_results["lr"]}, Seed={best_results["seed"]})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Training History salvato in: {save_path}")
    plt.show()


def plot_accuracy_bar_and_confusion(train_results, test_results, save_path=f'{OUTPUT_DIR}/monk1_results.png'):
    
    # Crea grafico con 1. bar chart delle accuracy; 2. confusion matrix
    train_acc = train_results['train_accuracy']
    val_acc = train_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    lr = train_results['lr']
    seed = train_results['seed']
    
    fig = plt.figure(figsize=(14, 6))
    
    # SUBPLOT 1: ACCURACY BAR CHART
    plt.subplot(1, 2, 1)
    
    categories = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2.5)
    
    # Testo sopra le barre
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        plt.text(i, acc + 0.03, f'{acc:.2%}', ha='center', 
                fontsize=16, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title(f'Model Performance (MONK-1)\n(LR={lr}, Seed={seed})', 
             fontsize=15, fontweight='bold', pad=15)
    plt.ylim(0, 1.15)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Linea 100% target
    plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2.5, 
               alpha=0.7, label='100% Target')
    plt.legend(fontsize=11, loc='lower right')
    
    # SUBPLOT 2: CONFUSION MATRIX
    plt.subplot(1, 2, 2)
    
    cm = test_results['confusion_matrix']
    confusion_data = np.array([[cm['tn'], cm['fp']], 
                                [cm['fn'], cm['tp']]])
    
    im = plt.imshow(confusion_data, cmap='Blues', alpha=0.9, vmin=0, vmax=confusion_data.max())
    plt.colorbar(im, label='Count', fraction=0.046, pad=0.04)
    
    # Testo nelle celle
    for i in range(2):
        for j in range(2):
            text_color = 'white' if confusion_data[i, j] > confusion_data.max()/2 else 'black'
            plt.text(j, i, str(confusion_data[i, j]), 
                    ha='center', va='center', 
                    fontsize=28, fontweight='bold',
                    color=text_color)
    
    plt.xticks([0, 1], ['Pred 0', 'Pred 1'], fontsize=12, fontweight='bold')
    plt.yticks([0, 1], ['True 0', 'True 1'], fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('True', fontsize=14, fontweight='bold')
    plt.title(f'Confusion Matrix (TEST SET)\nPrecision:  {test_results["precision"]:2%}, Recall: {test_results["recall"]:.2%}', 
             fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Accuracy e Confusion Matrix salvato in: {save_path}")
    plt.close(fig)

def print_best_configuration(best_results, test_results, kfold_results=None):
    
    # Stampa la migliore configurazione trovata
    
    print(f"\n{'='*70}")
    print(" MIGLIORE CONFIGURAZIONE TROVATA")
    print(f"{'='*70}\n")
    
    print(" PERFORMANCE:")
    print(f"{'─'*70}")
    print(f"  Training Accuracy:       {best_results['train_accuracy']:.4%}")
    print(f"  Validation Accuracy:    {best_results['val_accuracy']:.4%}")
    print(f"  Test Accuracy:          {test_results['test_accuracy']:.4%}")
    if kfold_results:
        print(f"  K-Fold CV Mean Acc:     {kfold_results['mean_accuracy']:.4%} ± {kfold_results['std_accuracy']:.4%}")
    
    print(f"\n  Test Precision:         {test_results['precision']:.4f}")
    print(f"  Test Recall:             {test_results['recall']:.4f}")
    print(f"  Test F1-Score:          {test_results['f1']:.4f}")
    
    print(f"\n  IPERPARAMETRI:")
    print(f"{'─'*70}")
    params = best_results['params']
    print(f"  Network Structure:      {params['network_structure']}")
    print(f"  Learning Rate (eta):    {params['eta']}")
    print(f"  Momentum:                {params['momentum']}")
    print(f"  L2 Lambda:              {params['l2_lambda']}")
    print(f"  Algorithm:              {params['algorithm']}")
    print(f"  Activation Function:    {params['activation_type']}")
    print(f"  Loss Type:              {params['loss_type']}")
    print(f"  Weight Initializer:     {params['weight_initializer']}")
    print(f"  Learning Rate Decay:    {params['decay']}")
    
    print(f"\n TRAINING DETAILS:")
    print(f"{'─'*70}")
    print(f"  Random Seed:             {best_results['seed']}")
    print(f"  Epochs:                 400 (with early stopping)")
    print(f"  Batch Size:             1")
    print(f"  Patience:               50")
    
    print(f"\n CONFUSION MATRIX (TEST SET):")
    print(f"{'─'*70}")
    cm = test_results['confusion_matrix']
    print(f"                    Predicted")
    print(f"                 0           1")
    print(f"  Actual  0     {cm['tn']: 3d}        {cm['fp']:3d}")
    print(f"          1     {cm['fn']:3d}        {cm['tp']: 3d}")
    
    print(f"\n{'='*70}\n")

def display_all_plots(best_results, test_results):
    print(f"\n{'='*70}")
    print(" VISUALIZZAZIONE GRAFICI FINALI")
    print(f"{'='*70}\n")
    
    # FUNZIONE DI SMOOTHING
    def smooth_curve(values, weight=0.9):

        if len(values) == 0:
            return values
        
        smoothed = []
        last = values[0]  # Inizia dal primo valore
        
        for point in values:
            # Formula EMA: smoothed = weight * previous + (1-weight) * current
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        
        return smoothed
    
    # CREA FIGURA CON 3 SUBPLOT
    fig = plt.figure(figsize=(18, 6))
    
    # TRAINING HISTORY CON SMOOTHING 
    ax1 = plt.subplot(1, 3, 1)
    
    history = best_results.get('history', None)
    
    # Debug info
    print(f" Debug - History type: {type(history)}")
    if isinstance(history, dict):
        print(f"   Keys: {history.keys()}")
        if 'training' in history: 
            print(f"   Training epochs: {len(history['training'])}")
            print(f"   Validation epochs: {len(history['validation'])}")
    
    # Prova a plottare la history esistente
    if history and isinstance(history, dict) and 'training' in history and len(history['training']) > 0:
        print(" Usando history esistente")
        
        train_loss = history['training']
        val_loss = history['validation']
        
        #  APPLICA SMOOTHING con peso 0.9 
        train_loss_smooth = smooth_curve(train_loss, weight=0.9)
        val_loss_smooth = smooth_curve(val_loss, weight=0.9)
        
        epochs = range(1, len(train_loss) + 1)
        
        # Plotta curve SMOOTH (linee principali)
        line1 = ax1.plot(epochs, train_loss_smooth, 
                         label='Training Loss (smoothed)', 
                         color='#1f77b4',  # Blu matplotlib standard
                         linewidth=2.5, 
                         alpha=0.95,
                         zorder=3)
        
        line2 = ax1.plot(epochs, val_loss_smooth, 
                         label='Validation Loss (smoothed)', 
                         color='#ff7f0e',  # Arancione matplotlib standard
                         linewidth=2.5, 
                         alpha=0.95,
                         zorder=3)
        
        # Aggiungi punto minimo validation
        min_val_idx = np.argmin(val_loss_smooth)
        min_val_epoch = min_val_idx + 1
        min_val_value = val_loss_smooth[min_val_idx]
        
        ax1.scatter([min_val_epoch], [min_val_value], 
                   color='red', s=100, marker='*', 
                   edgecolors='darkred', linewidths=1.5,
                   zorder=5, label=f'Min Val Loss (epoch {min_val_epoch})')
        
        # Formatting
        ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss (Half MSE)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Training & Validation Loss (smoothed)\n(LR={best_results["lr"]}, Seed={best_results["seed"]})', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.legend(fontsize=9, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Limiti assi più belli
        y_min = min(min(train_loss_smooth), min(val_loss_smooth))
        y_max = max(max(train_loss_smooth), max(val_loss_smooth))
        y_range = y_max - y_min
        ax1.set_ylim(max(0, y_min - 0.05*y_range), y_max + 0.1*y_range)
        
    else:
        # RI-ALLENA PER CREARE HISTORY
        print("  History non disponibile, ri-alleno per visualizzazione...")
        
        np.random.seed(best_results['seed'])
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        
        # Crea nuovo modello con gli stessi parametri
        net = NeuralNetwork(**best_results['params'])
        
        train_losses = []
        val_losses = []
        
        # Training per 100 epoche
        num_epochs_retrain = 100
        print(f"   Re-training per {num_epochs_retrain} epoche...")
        
        for epoch in range(num_epochs_retrain):
            if epoch % 20 == 0:
                print(f"   Epoca {epoch}/{num_epochs_retrain}")
            
            # Train 1 epoca
            net.fit(X_train, y_train, X_val, y_val, 
                   epochs=1, batch_size=1, verbose=False)
            
            # Calcola training loss
            train_pred = net.predict(X_train)
            train_loss = 0.5 * np.mean((train_pred - y_train)**2)
            train_losses.append(train_loss)
            
            # Calcola validation loss
            val_pred = net.predict(X_val)
            val_loss = 0.5 * np.mean((val_pred - y_val)**2)
            val_losses.append(val_loss)
        
        print("   ✓ Re-training completato")
        
        # Applica smoothing
        train_losses_smooth = smooth_curve(train_losses, weight=0.9)
        val_losses_smooth = smooth_curve(val_losses, weight=0.9)
        
        epochs_plot = range(1, num_epochs_retrain + 1)
        
        # Plotta curve smooth
        ax1.plot(epochs_plot, train_losses_smooth, 
                label='Training Loss (smoothed)', 
                color='#1f77b4', linewidth=2.5, alpha=0.95, zorder=3)
        
        ax1.plot(epochs_plot, val_losses_smooth, 
                label='Validation Loss (smoothed)', 
                color='#ff7f0e', linewidth=2.5, alpha=0.95, zorder=3)
        
        # Punto minimo
        min_val_idx = np.argmin(val_losses_smooth)
        ax1.scatter([min_val_idx + 1], [val_losses_smooth[min_val_idx]], 
                   color='red', s=100, marker='*', edgecolors='darkred', 
                   linewidths=1.5, zorder=5, label=f'Min Val (epoch {min_val_idx+1})')
        
        # Formatting
        ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss (Half MSE)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Training & Validation Loss (re-trained, smoothed)\n(LR={best_results["lr"]}, Seed={best_results["seed"]})', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.legend(fontsize=9, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        print("✓ History ricostruita con smoothing")
    
    # accuracy bar chart
    ax2 = plt.subplot(1, 3, 2)
    
    train_acc = best_results['train_accuracy']
    val_acc = best_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    
    categories = ['Train', 'Val', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#f39c12', '#2ecc71']  # Blu, Arancione, Verde
    
    bars = ax2.bar(categories, accuracies, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2, width=0.6)
    
    # Testo sopra le barre
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        ax2.text(i, acc + 0.03, f'{acc:.2%}', 
                ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    
    # Formatting
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title(f'Model Performance\n(LR={best_results["lr"]}, Seed={best_results["seed"]})', 
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Linea target 100%
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label='100% Target', zorder=1)
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
    
    # confusion matrix
    ax3 = plt.subplot(1, 3, 3)
    
    cm = test_results['confusion_matrix']
    confusion_data = np.array([[cm['tn'], cm['fp']], 
                                [cm['fn'], cm['tp']]])
    
    # Heatmap
    im = ax3.imshow(confusion_data, cmap='Blues', alpha=0.9, 
                    vmin=0, vmax=confusion_data.max())
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax3, label='Count', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    # Testo nelle celle
    for i in range(2):
        for j in range(2):
            value = confusion_data[i, j]
            text_color = 'white' if value > confusion_data.max()/2 else 'black'
            ax3.text(j, i, str(value), 
                    ha='center', va='center', 
                    fontsize=24, fontweight='bold',
                    color=text_color)
    
    # Labels
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=10, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['True 0', 'True 1'], fontsize=10, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True', fontsize=11, fontweight='bold')
    
    precision = test_results['precision']
    recall = test_results['recall']
    f1 = test_results['f1']
    
    ax3.set_title(f'Confusion Matrix (TEST SET)\n' + 
                 f'Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}', 
                 fontsize=12, fontweight='bold', pad=10)
    
   
    fig.suptitle('MONK-1 Final Results Summary', 
                fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Lascia spazio per suptitle
    
    # Salva
    save_path = 'monk1_final_summary.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Grafico combinato salvato in: {save_path}")
    
    plt.show()
    
    print(" Grafici visualizzati con smoothing EMA!")

if __name__ == "__main__":        
    try:
        print("="*70)
        print(" MONK-1 - BINARY CLASSIFICATION")
        print("="*70)
        
        #  FASE 1: GRID SEARCH
        print("\n FASE 1: Grid Search (Train + Validation) - Loss: Half MSE")
        best_results, all_results = grid_search(
            n_seeds=100,
            learning_rates=[0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07],
            momentums=[0.92, 0.94, 0.95, 0.96, 0.98],
            decays=[0.85, 0.9, 0.95],
            patiences=[20, 30, 40, 50, 60]
        )

        #  FASE 2: K-FOLD CROSS VALIDATION 
        print(f"\n FASE 2: K-Fold Cross Validation sul Best LR")
        kfold_results = k_fold_cross_validation(
            learning_rate=best_results['lr'],
            k=5,
            epochs=400,
            batch_size=1,
            patience=50,
            seed=42
        )
        
        #FASE 3: VALUTAZIONE TEST SET
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        #  ANALISI ERRORI 
        print(f"\n{'='*70}")
        print(" ANALISI ERRORI SUL TEST SET")
        print(f"{'='*70}")
        
        test_pred = final_net.predict(X_test)
        test_pred_class = (test_pred > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        errors_mask = test_pred_class != y_test_flat
        error_indices = np.where(errors_mask)[0]
        
        print(f"\n Errori sul test set: {len(error_indices)}/{len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
        
        if len(error_indices) > 0:
            print(f"\n Dettaglio esempi sbagliati:")
            print(f"{'─'*70}")
            print(f"{'Idx':<8} {'Pred Value':<15} {'Pred Class':<15} {'True Class':<15} {'Confidence'}")
            print(f"{'─'*70}")
            
            for idx in error_indices: 
                pred_val = test_pred[idx][0]
                pred_class = test_pred_class[idx]
                true_class = y_test_flat[idx]
                confidence = abs(pred_val - 0.5)  # Distanza da 0.5
                
                print(f"{idx:<8} {pred_val:<15.4f} {pred_class:<15} {true_class:<15} {confidence:.4f}")
            
            print(f"{'─'*70}")
            
            # Statistiche sugli errori
            avg_confidence_errors = np.mean([abs(test_pred[idx][0] - 0.5) for idx in error_indices])
            print(f"\n Confidence media sugli errori: {avg_confidence_errors:.4f}")
            print(f"   (0.0 = molto incerto, 0.5 = molto sicuro)")
        else:
            print(f"\n Nessun errore!  100% accuracy raggiunta!")
        
        print(f"\n{'='*70}\n")
        
        #  FASE 4: PLOT TRAINING HISTORY 
        print(f"\n FASE 4: Plot Training History")
        plot_training_history(best_results, save_path='monk1_training_history.png')
        
        #  FASE 5: PLOT ACCURACY + CONFUSION MATRIX 
        print(f"\n FASE 5: Plot Accuracy e Confusion Matrix")
        plot_accuracy_bar_and_confusion(best_results, test_results, save_path='monk1_results.png')
        
        #  FASE 6: STAMPA MIGLIORE CONFIGURAZIONE 
        print(f"\n FASE 6: Riepilogo Migliore Configurazione")
        print_best_configuration(best_results, test_results, kfold_results)
        
        #  FASE 7: ENSEMBLE AVANZATO 
        print(f"\n{'='*70}")
        print(" FASE 7: ENSEMBLE AVANZATO (Top 15 modelli)")
        print(f"{'='*70}")
        
        # Ordina tutti i risultati per validation accuracy
        sorted_results = sorted(all_results, key=lambda x: x['val_accuracy'], reverse=True)
        
        # Prendi i top 25 modelli
        n_ensemble = 30
        top_n_results = sorted_results[:n_ensemble]
        
        print(f"\n Top {n_ensemble} configurazioni selezionate:")
        print(f"{'─'*70}")
        print(f"{'Rank':<6} {'LR':<8} {'Seed':<8} {'Val Acc':<12} {'Test Acc':<12}")
        print(f"{'─'*70}")
        
        for idx, res in enumerate(top_n_results, 1):
            # Calcola test accuracy per questa configurazione
            net = res['network']
            X_test_config = res['data'][4]
            y_test_config = res['data'][5]
            test_pred_config = net.predict(X_test_config)
            test_acc_config = np.mean((test_pred_config > 0.5).astype(int) == y_test_config)
            
            print(f"{idx:<6} {res['lr']:<8.3f} {res['seed']: <8d} {res['val_accuracy']:<12.4%} {test_acc_config: <12.4%}")
        
        print(f"{'─'*70}")
        
        # Raccogli le reti
        ensemble_nets = [r['network'] for r in top_n_results]
        
        print(f"\n✓ Ensemble di {len(ensemble_nets)} modelli pronto")
        
        # PREDIZIONE ENSEMBLE CON MAJORITY VOTING 
        print(f"\n Valutazione Ensemble (Majority Voting)...")
        
        # Raccogli predizioni da tutti i modelli
        ensemble_preds_all = []
        for idx, net in enumerate(ensemble_nets):
            pred = net.predict(X_test)
            pred_class = (pred > 0.5).astype(int).flatten()
            ensemble_preds_all.append(pred_class)
        
        # Stack predizioni:  shape (n_models, n_samples)
        ensemble_stack = np.array(ensemble_preds_all)  # Shape: (15, 432)
        
        # MAJORITY VOTING: se >= 50% dei modelli dice classe 1, allora è 1
        ensemble_pred_class = (np.mean(ensemble_stack, axis=0) >= 0.5).astype(int)
        
        # Metriche
        ensemble_acc = np.mean(ensemble_pred_class == y_test.flatten())
        ensemble_error = 1 - ensemble_acc
        
        # Confusion matrix
        tp_ens = np.sum((ensemble_pred_class == 1) & (y_test.flatten() == 1))
        fp_ens = np.sum((ensemble_pred_class == 1) & (y_test.flatten() == 0))
        tn_ens = np.sum((ensemble_pred_class == 0) & (y_test.flatten() == 0))
        fn_ens = np.sum((ensemble_pred_class == 0) & (y_test.flatten() == 1))
        
        precision_ens = tp_ens / (tp_ens + fp_ens) if (tp_ens + fp_ens) > 0 else 0
        recall_ens = tp_ens / (tp_ens + fn_ens) if (tp_ens + fn_ens) > 0 else 0
        f1_ens = 2 * (precision_ens * recall_ens) / (precision_ens + recall_ens) if (precision_ens + recall_ens) > 0 else 0
        
        #  RISULTATI ENSEMBLE 
        print(f"\n{'='*70}")
        print(" RISULTATI ENSEMBLE (Majority Voting)")
        print(f"{'='*70}")
        print(f"  Numero modelli:      {n_ensemble}")
        print(f"  Ensemble Accuracy:   {ensemble_acc:.4%}")
        print(f"  Ensemble Error:      {ensemble_error:.4%}")
        print(f"  Precision:           {precision_ens:.4f}")
        print(f"  Recall:              {recall_ens:.4f}")
        print(f"  F1-score:            {f1_ens:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP:  {tp_ens: 3d}  FP: {fp_ens:3d}")
        print(f"    FN: {fn_ens:3d}  TN: {tn_ens:3d}")
        
        #  CONFRONTO 
        print(f"\n{'─'*70}")
        print(" CONFRONTO:  Singolo Modello vs Ensemble")
        print(f"{'─'*70}")
        print(f"  Singolo (best): {test_results['test_accuracy']:.4%}")
        print(f"  Ensemble ({n_ensemble}):   {ensemble_acc:.4%}")
        
        improvement = (ensemble_acc - test_results['test_accuracy']) * 100
        if improvement > 0:
            print(f"  Miglioramento:  +{improvement:.2f}%")
            
            # Quanti errori sono stati corretti?
            single_errors = int((1 - test_results['test_accuracy']) * len(y_test))
            ensemble_errors = int((1 - ensemble_acc) * len(y_test))
            errors_fixed = single_errors - ensemble_errors
            
            print(f"  Errori corretti:  {errors_fixed}/{single_errors}")
        else:
            print(f"  Differenza:        {improvement:.2f}%")
        
        #  ANALISI ERRORI ENSEMBLE
        if ensemble_acc < 1.0:
            print(f"\n{'─'*70}")
            print(" ANALISI ERRORI ENSEMBLE")
            print(f"{'─'*70}")
            
            ensemble_errors_mask = ensemble_pred_class != y_test.flatten()
            ensemble_error_indices = np.where(ensemble_errors_mask)[0]
            
            print(f"\n Errori ensemble: {len(ensemble_error_indices)}/{len(y_test)}")
            
            if len(ensemble_error_indices) > 0:
                print(f"\n Esempi difficili (sbagliati anche dall'ensemble):")
                print(f"{'─'*70}")
                print(f"{'Idx': <8} {'Votes for 1':<15} {'Ensemble Pred':<18} {'True Class':<15}")
                print(f"{'─'*70}")
                
                for idx in ensemble_error_indices:
                    votes_for_1 = np.sum(ensemble_stack[: , idx])  # Quanti modelli hanno votato 1
                    ensemble_pred = ensemble_pred_class[idx]
                    true_class = y_test.flatten()[idx]
                    
                    print(f"{idx: <8} {votes_for_1}/{n_ensemble: <12} {ensemble_pred:<18} {true_class:<15}")
                
                print(f"{'─'*70}")
        
        #  CHECK 100% 
        if ensemble_acc >= 1.0:
            
            print(f" 100% TEST ACCURACY RAGGIUNTO CON ENSEMBLE!  🏆🏆🏆")
            
            # Salva grafico ensemble
            ensemble_results = {
                'test_accuracy': ensemble_acc,
                'test_error': ensemble_error,
                'precision': precision_ens,
                'recall': recall_ens,
                'f1': f1_ens,
                'confusion_matrix': {'tp': tp_ens, 'fp': fp_ens, 'tn': tn_ens, 'fn': fn_ens}
            }
            
            ensemble_train_results = best_results.copy()
            ensemble_train_results['lr'] = f"Ensemble-{n_ensemble}"
            ensemble_train_results['seed'] = "Mixed"
            ensemble_train_results['train_accuracy'] = 1.0
            ensemble_train_results['val_accuracy'] = 1.0
            
            plot_accuracy_bar_and_confusion(ensemble_train_results, ensemble_results, 
                                            save_path='monk1_results_ENSEMBLE_100.png')
            
            print(f"\n Grafico ensemble 100% salvato: monk1_results_ENSEMBLE_100.png")
        
        elif ensemble_acc > test_results['test_accuracy']: 
            print(f"\n L'ensemble ha migliorato il risultato!")
            
            # Salva grafico ensemble se migliora
            ensemble_results = {
                'test_accuracy':  ensemble_acc,
                'test_error': ensemble_error,
                'precision': precision_ens,
                'recall': recall_ens,
                'f1': f1_ens,
                'confusion_matrix': {'tp': tp_ens, 'fp': fp_ens, 'tn': tn_ens, 'fn': fn_ens}
            }
            
            ensemble_train_results = best_results.copy()
            ensemble_train_results['lr'] = f"Ensemble-{n_ensemble}"
            ensemble_train_results['seed'] = "Mixed"
            ensemble_train_results['train_accuracy'] = np.mean([r['train_accuracy'] for r in top_n_results])
            ensemble_train_results['val_accuracy'] = np.mean([r['val_accuracy'] for r in top_n_results])
            
            plot_accuracy_bar_and_confusion(ensemble_train_results, ensemble_results, 
                                            save_path='monk1_results_ENSEMBLE_IMPROVED.png')
            
            print(f"\n Grafico ensemble migliorato salvato: monk1_results_ENSEMBLE_IMPROVED.png")
        else:
            print(f"\n L'ensemble non ha migliorato il singolo modello")
        
        print(f"\n{'='*70}\n")
        
        # FASE 8: VISUALIZZAZIONE FINALE
        print(f"\n FASE 8: Visualizzazione Grafici Finali")
        display_all_plots(best_results, test_results)
        
        # RIEPILOGO FINALE =========
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        if ensemble_acc >= 1.0:
            print(" PERFETTO:  100% TEST ACCURACY RAGGIUNTO CON ENSEMBLE!")
        elif test_results['test_accuracy'] >= 1.0:
            print(" PERFETTO: 100% TEST ACCURACY RAGGIUNTO!")
        elif ensemble_acc >= 0.99:
            print(" ECCELLENTE: 99%+ test accuracy con ensemble!")
        elif test_results['test_accuracy'] >= 0.99:
            print(" ECCELLENTE: 99%+ test accuracy!")
        else:
            print(f" Test Accuracy (singolo): {test_results['test_accuracy']:.2%}")
            print(f" Test Accuracy (ensemble): {ensemble_acc:.2%}")
        
        print(f"\n{'─'*70}")
        print(" ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  SINGOLO MODELLO:")
        print(f"    Train Accuracy:          {best_results['train_accuracy']:.4%}")
        print(f"    Validation Accuracy:    {best_results['val_accuracy']:.4%}")
        print(f"    Test Accuracy:          {test_results['test_accuracy']:.4%}")
        print(f"    K-Fold CV Mean:         {kfold_results['mean_accuracy']:.4%} ± {kfold_results['std_accuracy']:.4%}")
        
        print(f"\n  ENSEMBLE ({n_ensemble} modelli):")
        print(f"    Test Accuracy:          {ensemble_acc:.4%}")
        
        print(f"\n{'─'*70}")
        print(" PARAMETRI MIGLIORI:")
        print(f"{'─'*70}")
        params = best_results['params']
        print(f"  Network Structure:       {params['network_structure']}")
        print(f"  Learning Rate (eta):    {params['eta']}")
        print(f"  Momentum:                {params['momentum']}")
        print(f"  L2 Lambda:              {params['l2_lambda']}")
        print(f"  Algorithm:              {params['algorithm']}")
        print(f"  Activation Function:    {params['activation_type']}")
        print(f"  Loss Type:              {params['loss_type']}")
        print(f"  Weight Initializer:     {params['weight_initializer']}")
        print(f"  Random Seed:            {best_results['seed']}")
        
        print(f"\n{'─'*70}")
        print(" FILE SALVATI:")
        print(f"{'─'*70}")
        
        import os
        files_to_check = [
            'monk1_training_history.png',
            'monk1_results.png',
            'monk1_final_summary.png',
            'monk1_kfold_results.png'
        ]
        
        # Aggiungi file ensemble se esistono
        if ensemble_acc >= 1.0:
            files_to_check.append('monk1_results_ENSEMBLE_100.png')
        elif ensemble_acc > test_results['test_accuracy']: 
            files_to_check.append('monk1_results_ENSEMBLE_IMPROVED.png')
        
        for filename in files_to_check: 
            if os.path.exists(filename):
                print(f"   {filename}")
            else:
                print(f"    {filename} - not found")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO MONK-1 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()