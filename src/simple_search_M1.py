import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk1


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
        'network_structure': [17, 5, 1],
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


def grid_search_lr(n_seeds_per_lr=5, learning_rates=None):
    """
    Grid search su learning rate
    SELEZIONE BASATA SU VALIDATION ACCURACY (non test!)
    """
    if learning_rates is None: 
        learning_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_val_acc = 0
    best_results = None
    all_results = []
    
    total_runs = len(learning_rates) * n_seeds_per_lr
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH:  {len(learning_rates)} LRs × {n_seeds_per_lr} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"🎯 TESTING LEARNING RATE: {lr}")
        print(f"{'='*70}")
        
        for seed_idx in range(n_seeds_per_lr):
            current_run += 1
            seed = seed_idx * 123 + int(lr * 1000)
            
            print(f"\n Run {current_run}/{total_runs} - LR={lr}, Seed={seed}", end=" → ")
            
            results = _monk1_test(learning_rate=lr, seed=seed, verbose=False)
            all_results.append(results)
            
            print(f"Train:  {results['train_accuracy']:.2%}, Val: {results['val_accuracy']:.2%}")
            
            if results['val_accuracy'] > best_val_acc: 
                best_val_acc = results['val_accuracy']
                best_results = results
                print(f"    NUOVO BEST VAL ACC: {best_val_acc:.2%}")
            
            if results['val_accuracy'] >= 1.0:
                print(f"\n 100% VALIDATION ACCURACY! Fermata anticipata.")
                break
        
        if best_val_acc >= 1.0:
            print(f"\n Obiettivo raggiunto! Interrompo la ricerca.")
            break
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE (basata su VALIDATION)")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {best_results['val_accuracy']:.4%}")
    print(f"Train Accuracy:       {best_results['train_accuracy']:.4%}")
    print(f"Best LR:             {best_results['lr']}")
    print(f"Best Seed:           {best_results['seed']}")
    
    return best_results, all_results


def evaluate_on_test_set(net, X_test, y_test):
    """
    Valuta il modello finale sul TEST SET (UNA SOLA VOLTA!)
    """
    print(f"\n{'='*70}")
    print(f"🎯 VALUTAZIONE FINALE SUL TEST SET")
    print(f"{'='*70}")
    
    test_pred = net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_acc = np.mean(test_pred_class == y_test)
    test_error = 1 - test_acc

    # CALCOLO HALF MSE SUL TEST SET
    test_half_mse = (1/2) * np.mean((test_pred - y_test)**2)
    
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
    print(f"  Test Half MSE:  {test_half_mse:.6f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print(f"\n Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN:  {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error': test_error,
        'test_half_mse': test_half_mse,
        'precision': precision,
        'recall': recall,
        'f1':  f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def plot_results(train_results, test_results, save_path='monk1_performance.png'):
    """
    Crea grafico con Model Performance (bar chart) + Confusion Matrix
    """
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
    plt.title(f'Confusion Matrix (TEST SET)\nPrecision: {test_results["precision"]:.2%}, Recall: {test_results["recall"]:.2%}', 
             fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Model Performance salvato in: {save_path}")
    plt.show()


def plot_loss_epochs(all_results, dataset_name='MONK-1', save_path='monk1_loss_epochs.png'):
    """
    Crea grafici per: 
    1. Loss (Half MSE) vs Epoche per Training e Validation
    2. Accuracy vs Epoche per Training e Validation
    """
    from collections import defaultdict
    
    print(f"\n{'='*70}")
    print(f"CREAZIONE GRAFICI LOSS E ACCURACY VS EPOCHE")
    print(f"{'='*70}")
    
    high_perf_results = [r for r in all_results if r['val_accuracy'] >= 0.95]

    if len(high_perf_results) < 5:
        print(f"  Solo {len(high_perf_results)} modelli con val_acc >= 95%.  Uso tutti i risultati.")
        results_to_use = all_results
    else:
        print(f" Usando {len(high_perf_results)} modelli con val_acc >= 95%")
        results_to_use = high_perf_results

    # RE-TRAINING DI ALCUNI MODELLI CON TRACKING
    print("Re-training di modelli selezionati con epoch tracking...")
    
    all_curves = []
    num_runs = min(15, len(results_to_use))  # Limita a 15 run
    
    for idx in range(num_runs):
        result = results_to_use[idx]
        print(f"\rProcessing run {idx+1}/{num_runs}.. .", end="")
        
        X_train, y_train, X_val, y_val, X_test, y_test = result['data']
        
        # Seed per riproducibilità
        np.random.seed(result['seed'])
        
        # Ricrea la rete
        params = result['params'].copy()
        net = NeuralNetwork(**params)
        
        train_losses_run = []
        val_losses_run = []
        train_accs_run = []
        val_accs_run = []
        epochs_run = []
        
        max_epochs = 400
        batch_size = 1
        
        for epoch in range(max_epochs):
            print(f"\nRe-training epoch {epoch}.")
            try:
                net.fit(X_train, y_train, X_val, y_val, epochs=epoch, batch_size=batch_size, verbose=False)
            except:  
                break
            
            # Training metrics
            train_pred = net.predict(X_train)
            train_loss = (1/2) * np.mean((train_pred - y_train)**2)  # HALF MSE LOSS
            train_pred_class = (train_pred > 0.5).astype(int)
            train_acc = np.mean(train_pred_class == y_train)
            
            # Validation metrics
            val_pred = net.predict(X_val)
            val_loss = (1/2) * np.mean((val_pred - y_val)**2)  # HALF MSE LOSS
            val_pred_class = (val_pred > 0.5).astype(int)
            val_acc = np.mean(val_pred_class == y_val)
            
            epochs_run.append(epoch + 1)
            train_losses_run.append(train_loss)
            val_losses_run.append(val_loss)
            train_accs_run.append(train_acc)
            val_accs_run.append(val_acc)
        
        all_curves.append((epochs_run, train_losses_run, val_losses_run, train_accs_run, val_accs_run))
    
    print("\nRe-training completato")
    
    # SMOOTHING DELLE CURVE
    def smooth_curve(values, weight=0.85):
        """Exponential moving average"""
        smoothed = []
        last = values[0] if values else 0
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # CALCOLA MEDIE PER EPOCA
    max_len = max(len(curve[0]) for curve in all_curves) if all_curves else 100
    
    train_loss_means = []
    val_loss_means = []
    train_acc_means = []
    val_acc_means = []
    
    for epoch_idx in range(max_len):
        train_loss_vals = []
        val_loss_vals = []
        train_acc_vals = []
        val_acc_vals = []
        
        for epochs_run, train_losses_run, val_losses_run, train_accs_run, val_accs_run in all_curves: 
            if epoch_idx < len(train_losses_run):
                train_loss_vals.append(train_losses_run[epoch_idx])
                val_loss_vals.append(val_losses_run[epoch_idx])
                train_acc_vals.append(train_accs_run[epoch_idx])
                val_acc_vals.append(val_accs_run[epoch_idx])
        
        if train_loss_vals:
            train_loss_means.append(np.mean(train_loss_vals))
            val_loss_means.append(np.mean(val_loss_vals))
            train_acc_means.append(np.mean(train_acc_vals))
            val_acc_means.append(np.mean(val_acc_vals))
    
    epochs_axis = range(1, len(train_loss_means) + 1)
    
    # CREAZIONE FIGURE CON 2 SUBPLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # SUBPLOT 1: LOSS (HALF MSE)
    if train_loss_means: 
        train_loss_smooth = smooth_curve(train_loss_means, weight=0.98)
        val_loss_smooth = smooth_curve(val_loss_means, weight=0.98)
        
        # Linee loss
        ax1.plot(epochs_axis, train_loss_smooth, label='Training Loss (Half MSE)', 
                color='#1F618D', linewidth=2.5, alpha=0.9, zorder=10)
        ax1.plot(epochs_axis, val_loss_smooth, label='Validation Loss (Half MSE)', 
                color='#E74C3C', linewidth=2.5, alpha=0.9, zorder=10)
        
        # Optimal complexity (minimo validation loss)
        min_val_loss_idx = np.argmin(val_loss_smooth)
        optimal_epoch_loss = min_val_loss_idx + 1
        optimal_loss = val_loss_smooth[min_val_loss_idx]
        
        ax1.axvline(x=optimal_epoch_loss, color='green', linestyle=':', 
                   linewidth=2.5, alpha=0.7, zorder=9,
                   label=f'Optimal: {optimal_epoch_loss} epochs')
        
        ax1.scatter([optimal_epoch_loss], [optimal_loss], color='green', 
                   s=250, marker='*', zorder=11, edgecolors='black', linewidths=2)
        
        # Annotazione
        y_max_loss = max(max(train_loss_smooth), max(val_loss_smooth))
        y_min_loss = min(min(train_loss_smooth), min(val_loss_smooth))
        y_range_loss = y_max_loss - y_min_loss if y_max_loss > y_min_loss else 1.0
        
        ax1.set_ylim(max(0, y_min_loss - 0.05 * y_range_loss), y_max_loss + 0.15 * y_range_loss)
    
    ax1.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Half MSE Loss', fontsize=13, fontweight='bold')
    ax1.set_title(f'Training & Validation Loss vs Epochs\n{dataset_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(0, max_len * 1.02)
    
    
    # SUBPLOT 2: ACCURACY
    if train_acc_means:
        train_acc_smooth = smooth_curve(train_acc_means, weight=0.98)
        val_acc_smooth = smooth_curve(val_acc_means, weight=0.98)
        
        # Linee accuracy
        ax2.plot(epochs_axis, train_acc_smooth, label='Training Accuracy', 
                color='#2874A6', linewidth=2.5, alpha=0.9, zorder=10)
        ax2.plot(epochs_axis, val_acc_smooth, label='Validation Accuracy', 
                color='#D35400', linewidth=2.5, alpha=0.9, zorder=10)
        
        # Optimal complexity (massima validation accuracy)
        max_val_acc_idx = np.argmax(val_acc_smooth)
        optimal_epoch_acc = max_val_acc_idx + 1
        optimal_acc = val_acc_smooth[max_val_acc_idx]
        
        ax2.axvline(x=optimal_epoch_acc, color='green', linestyle=':', 
                   linewidth=2.5, alpha=0.7, zorder=9,
                   label=f'Optimal:  {optimal_epoch_acc} epochs')
        
        ax2.scatter([optimal_epoch_acc], [optimal_acc], color='green', 
                   s=250, marker='*', zorder=11, edgecolors='black', linewidths=2)
        
        # Linea 100% target
        ax2.axhline(y=1.0, color='#27AE60', linestyle='--', linewidth=2, 
                   alpha=0.6, label='100% Target', zorder=8)
        
        # Range accuracy
        y_max_acc = min(1.05, max(max(train_acc_smooth), max(val_acc_smooth)) + 0.05)
        y_min_acc = max(0, min(min(train_acc_smooth), min(val_acc_smooth)) - 0.05)
        
        ax2.set_ylim(y_min_acc, y_max_acc)
    
    ax2.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'Training & Validation Accuracy vs Epochs\n{dataset_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(0, max_len * 1.02)
    
    # Formattazione asse y come percentuale
    from matplotlib.ticker import PercentFormatter
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()
    
     # Stampa statistiche finali
    if train_loss_means and train_acc_means:
        print(f"\n{'─'*70}")
        print(f" STATISTICHE FINALI (media su {num_runs} runs):")
        print(f"{'─'*70}")
        print(f"  Optimal Epoch (Loss):      {optimal_epoch_loss}")
        print(f"  Min Validation Loss:       {optimal_loss:.6f}")
        print(f"  Optimal Epoch (Accuracy):  {optimal_epoch_acc}")
        print(f"  Max Validation Accuracy:   {optimal_acc:.4%}")
        print(f"  Final Training Accuracy:   {train_acc_smooth[-1]:.4%}")
        print(f"  Final Validation Accuracy: {val_acc_smooth[-1]:.4%}")
        print(f"\n{'─'*70}")
        print(f" HALF MSE LOSS FINALE:")
        print(f"{'─'*70}")
        print(f"  Training Set Half MSE:     {train_loss_smooth[-1]:.6f}")
        print(f"  Validation Set Half MSE:   {val_loss_smooth[-1]:.6f}")


if __name__ == "__main__":  
    try:
        print("="*70)
        print(" MONK-1 - BINARY CLASSIFICATION")
        print("="*70)
        
        # FASE 1: Grid search
        print("\n FASE 1: Grid Search (Train + Validation)")
        best_results, all_results = grid_search_lr(
            n_seeds_per_lr=50,
            learning_rates=[0.005, 0.1, 0.15, 0.2, 0.25]
        )
        
        # FASE 2: LOSS VS EPOCHS
        print(f"\n FASE 2: Grafico Loss vs Epochs")
        plot_loss_epochs(all_results, 
                                  dataset_name='MONK-1',
                                  save_path='monk1_loss_epochs.png')
        
        # FASE 3: Valuta sul TEST SET (UNA SOLA VOLTA!)
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Model Performance + Confusion Matrix
        print(f"\n FASE 4: Grafico Model Performance")
        plot_results(best_results, test_results, save_path='monk1_performance.png')
        
        # FASE 5: Valuta sul TEST SET (UNA SOLA VOLTA!)
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Model Performance + Confusion Matrix
        print(f"\n FASE 4: Grafico Model Performance")
        plot_results(best_results, test_results, save_path='monk1_performance.png')
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        if test_results['test_accuracy'] >= 1.0:
            print(" PERFETTO:  100% TEST ACCURACY RAGGIUNTO!")
        elif test_results['test_accuracy'] >= 0.99:
            print("  ECCELLENTE: 99%+ test accuracy!")
        else:
            print(f" Test Accuracy: {test_results['test_accuracy']:.2%}")
        
        print(f"\n{'─'*70}")
        print(" ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Accuracy:          {best_results['train_accuracy']:.4%}")
        print(f"  Validation Accuracy:     {best_results['val_accuracy']:.4%}")
        print(f"  Test Accuracy (FINAL):   {test_results['test_accuracy']:.4%}")
        
        print(f"\n{'─'*70}")
        print(" HALF MSE LOSS SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Half MSE:          {best_results['train_loss']:.6f}")
        print(f"  Validation Half MSE:     {best_results['val_loss']:.6f}")
        print(f"  Test Half MSE (FINAL):   {test_results['test_half_mse']:.6f}")
        
        print(f"\n{'─'*70}")
        print(" PARAMETRI MIGLIORI:")
        print(f"{'─'*70}")
        print(f"  Learning Rate:   {best_results['lr']}")
        print(f"  Random Seed:     {best_results['seed']}")
        print(f"  Hidden Units:    {best_results['params']['network_structure'][1]}")
        print(f"  Momentum:        {best_results['params']['momentum']}")
        print(f"  Architecture:    {best_results['params']['network_structure']}")
        
        print(f"\n{'─'*70}")
        print(" METRICHE DETTAGLIATE (TEST SET):")
        print(f"{'─'*70}")
        print(f"  Accuracy:    {test_results['test_accuracy']:.4%}")
        print(f"  Precision:   {test_results['precision']:.4f}")
        print(f"  Recall:      {test_results['recall']:.4f}")
        print(f"  F1-score:    {test_results['f1']:.4f}")
        
        cm = test_results['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives  (TP): {cm['tp']}")
        print(f"    False Positives (FP): {cm['fp']}")
        print(f"    True Negatives  (TN): {cm['tn']}")
        print(f"    False Negatives (FN): {cm['fn']}")
        
        print(f"\n File salvati:")
        print(f"  - monk1_loss_epochs.png (loss e accuracy vs epochs)")
        print(f"  - monk1_performance.png (confusion matrix)")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO MONK-1 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e: 
        print(f"\n ERRORE:  {e}")
        import traceback
        traceback.print_exc()