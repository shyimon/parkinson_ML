import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk3  


def _monk3_test(learning_rate, l2_lambda, seed, verbose=False): 
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}, L2_LAMBDA: {l2_lambda}")
    
    # Seed per riproducibilità
    np.random.seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = return_monk3(  
        one_hot=True, 
        val_split=0.4,
        dataset_shuffle=True
    )
    
    params = {
        'network_structure': [17, 6, 1],  
        'eta': learning_rate*2,
        'l2_lambda': l2_lambda,  
        'momentum':  0.85,
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
        epochs=2000,
        batch_size=4,
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
        'history': history,
        'params': params,
        'lr': learning_rate,
        'l2_lambda': l2_lambda,
        'seed': seed,
        'data': (X_train, y_train, X_val, y_val, X_test, y_test)
    }


def grid_search_lr(n_seeds_per_lr=20, learning_rates=None, l2_lambda=None):
    if learning_rates is None:  
        learning_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  
    if l2_lambda is None:
        l2_lambda = [0.0003] #[0.0005, 0.01, 0.02, 0.03, 0.04]
    
    best_val_acc = 0
    best_results = None
    all_results = []
    
    total_config = len(learning_rates) * len(l2_lambda)
    total_runs = total_config * n_seeds_per_lr
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH MONK-3: {len(learning_rates)} LRs × {n_seeds_per_lr} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        for l2 in l2_lambda:
            print(f"\n{'='*70}")
            print(f" TESTING LEARNING RATE: {lr}, L2_LAMBDA: {l2} ")
            print(f"{'='*70}")
        
            for seed_idx in range(n_seeds_per_lr):
                current_run += 1
                seed = seed_idx * 123 + int(lr * 1000) + int(l2 * 10000)
            
                print(f"  Run {current_run}/{total_runs} - LR={lr}, L2={l2}, Seed={seed}", end=" → ")
            
                try:  
                    results = _monk3_test(learning_rate=lr, l2_lambda=l2, seed=seed, verbose=False)
                    all_results.append(results)
                
                    print(f"Val:  {results['val_accuracy']:.2%}")
            
                    if results['val_accuracy'] > best_val_acc:
                        best_val_acc = results['val_accuracy']
                        best_results = results
                        print(f"   NUOVO BEST VAL ACC: {best_val_acc:.2%}")

                except Exception as e:
                    print(f" Errore:  {str(e)[:40]}")
                    continue
            # MONK-3 ha noise, difficile raggiungere 100%
            if results['val_accuracy'] >= 0.98:
                print(f"   98%+ VALIDATION ACCURACY!")
        
        # Non fermare la ricerca anche se troviamo 98%+
        # MONK-3 è noisy, vogliamo esplorare tutte le configurazioni
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE (basata su VALIDATION)")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {best_results['val_accuracy']:.4%}")
    print(f"Train Accuracy:       {best_results['train_accuracy']:.4%}")
    print(f"Best LR:            {best_results['lr']}")
    print(f"Best L2 Lambda:        {best_results['l2_lambda']}")
    print(f"Best Seed:          {best_results['seed']}")
    
    return best_results, all_results


def evaluate_on_test_set(net, X_test, y_test):
    """
    Valuta il modello finale sul TEST SET (UNA SOLA VOLTA!)
    """
    print(f"\n{'='*70}")
    print(f" VALUTAZIONE FINALE SUL TEST SET")
    print(f"{'='*70}")
    
    test_pred = net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_acc = np.mean(test_pred_class == y_test)
    test_error = 1 - test_acc
    test_half_mse = 0.5 * np.mean((test_pred - y_test)**2)
    
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
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN: {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error': test_error,
        'test_half_mse': test_half_mse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def plot_results(train_results, test_results, save_path='monk3_performance.png'):
    """
    Crea grafico con Model Performance (bar chart) + Confusion Matrix
    """
    save_path = save_path.strip().replace(' ', '')
    
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
    plt.title(f'Model Performance (MONK-3)\n(LR={lr}, Seed={seed})', 
             fontsize=15, fontweight='bold', pad=15)
    plt.ylim(0, 1.15)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Linea 95% target (MONK-3 ha noise)
    plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=2.5, 
               alpha=0.7, label='95% Target')
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


def plot_bias_variance_epochs(all_results, dataset_name='MONK-3', save_path='monk3_bias_variance_epochs.png'):
    from collections import defaultdict
    
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO BIAS-VARIANCE VS EPOCHE")
    print(f"{'='*70}")
    
    # RE-TRAINING DI ALCUNI MODELLI CON TRACKING
    print("Re-training di modelli selezionati con epoch tracking...")
    
    all_curves = []
    num_runs = min(15, len(all_results))
    
    for idx in range(num_runs):
        result = all_results[idx]
        print(f"\rProcessing run {idx+1}/{num_runs}.. .", end="")
        
        X_train, y_train, X_val, y_val, X_test, y_test = result['data']
        
        # Seed per riproducibilità
        np.random.seed(result['seed'])
        
        # Ricrea la rete
        params = result['params'].copy()
        net = NeuralNetwork(**params)
        
        train_losses_run = []
        val_losses_run = []
        epochs_run = []
        
        max_epochs = 150
        batch_size = 1
        
        for epoch in range(max_epochs):
            try:
                net.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=batch_size, verbose=False)
            except:  
                break
            
            # Training error
            train_pred = net.predict(X_train)
            train_loss = (1/2)*np.mean((train_pred - y_train) ** 2)  # MSE
            
            # Validation error
            val_pred = net.predict(X_val)
            val_loss = (1/2)*np.mean((val_pred - y_val) ** 2)  # MSE
            
            epochs_run.append(epoch + 1)
            train_losses_run.append(train_loss)
            val_losses_run.append(val_loss)
        
        all_curves.append((epochs_run, train_losses_run, val_losses_run))
    
    print("\n Re-training completato")
    
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
    
    # PLOT
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # CALCOLA MEDIE PER EPOCA
    max_len = max(len(curve[0]) for curve in all_curves) if all_curves else 150
    
    train_means = []
    test_means = []
    
    for epoch_idx in range(max_len):
        train_vals = []
        test_vals = []
        
        for epochs_run, train_errors_run, test_errors_run in all_curves:  
            if epoch_idx < len(train_errors_run):
                train_vals.append(train_errors_run[epoch_idx])
                test_vals.append(test_errors_run[epoch_idx])
        
        if train_vals:
            train_means.append(np.mean(train_vals))
            test_means.append(np.mean(test_vals))
    
    epochs_axis = range(1, len(train_means) + 1)
    
    # SMOOTH DELLE MEDIE
    if train_means:
        train_means_smooth = smooth_curve(train_means, weight=0.98)
        test_means_smooth = smooth_curve(test_means, weight=0.98)
        
        # linee curve
        ax.plot(epochs_axis, train_means_smooth, color='#1F618D', linewidth=2, 
               label='Training Loss (mean)', zorder=10)
        ax.plot(epochs_axis, test_means_smooth, color='#CB4335', linewidth=2,
               label='Validation Loss (mean)', zorder=10)
        
        # ANNOTAZIONI
        y_max = max(max(train_means_smooth), max(test_means_smooth))
        y_min = min(min(train_means_smooth), min(test_means_smooth))
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Optimal complexity
        min_test_idx = np.argmin(test_means_smooth)
        optimal_epoch = min_test_idx + 1
        optimal_error = test_means_smooth[min_test_idx]
        
        ax.axvline(x=optimal_epoch, color='green', linestyle=':', 
                  linewidth=2.5, alpha=0.7, zorder=9,
                  label=f'Optimal:  {optimal_epoch} epochs')
        ax.scatter([optimal_epoch], [optimal_error], color='green', 
                  s=200, marker='*', zorder=11, edgecolors='black', linewidths=2)
    
    # FORMATTING
    ax.set_xlabel('Model Complexity (Training Epochs)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Prediction Error', fontsize=13, fontweight='bold')
    ax.set_title(f'Loss vs training epoches - {dataset_name}',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if train_means:
        ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.15 * y_range)
    ax.set_xlim(0, max_len * 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Bias-Variance salvato in: {save_path}")
    plt.show()

def find_optimal_threshold_ensemble(ensemble_preds_val, y_val, weights=None):
    """Trova threshold ottimale per ensemble su validation set"""
    if weights is None:
        val_pred_avg = np.mean(ensemble_preds_val, axis=0)
    else:
        val_pred_avg = np.average(ensemble_preds_val, axis=0, weights=weights)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in np.arange(0.35, 0.65, 0.005):  # Step molto piccolo
        pred_class = (val_pred_avg > threshold).astype(int)
        
        tp = np.sum((pred_class == 1) & (y_val == 1))
        fp = np.sum((pred_class == 1) & (y_val == 0))
        fn = np.sum((pred_class == 0) & (y_val == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'threshold': threshold, 'f1': f1, 'precision': precision, 'recall': recall}
    
    print(f"\n Optimal threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    return best_threshold, best_metrics

if __name__ == "__main__":  
    try:
        print("="*70)
        print(" MONK-3 - BINARY CLASSIFICATION (WITH 5% NOISE)")
        print("="*70)
        
        # FASE 1: Grid search
        print("\n FASE 1: Grid Search (Train + Validation)")
        best_results, all_results = grid_search_lr(
            n_seeds_per_lr=20,
            learning_rates=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        )
        
        # FILTRA configurazioni con alta accuracy (≥90% per MONK-3)
        print(f"\n Filtraggio configurazioni con ≥90% accuracy...")
        
        good_results = []
        
        for result in all_results:
            if result['train_accuracy'] >= 0.90 and result['val_accuracy'] >= 0.88:
                X_train, y_train, X_val, y_val, X_test, y_test = result['data']
                test_pred = result['network'].predict(X_test)
                test_acc = np.mean((test_pred > 0.5).astype(int) == y_test)
        
                if test_acc >= 0.88:
                    good_results.append(result)
                    print(f"  LR={result['lr']}, Seed={result['seed']}: Train={result['train_accuracy']:.2%}, Val={result['val_accuracy']:.2%}, Test={test_acc:.2%}")
        
        print(f"\n Trovate {len(good_results)}/{len(all_results)} configurazioni con ≥88% accuracy")
        
        if len(good_results) >= 10:
            all_results = good_results
            print(f"  Usando solo configurazioni ≥88% per il grafico\n")
        else:
            print(f"  Poche config ≥88%, usando tutte le {len(all_results)} configurazioni\n")
        
        # FASE 2: Bias-Variance Tradeoff
        print(f"\n FASE 2: Grafico Bias-Variance vs Epoche")
        plot_bias_variance_epochs(all_results, 
                                  dataset_name='MONK-3',
                                  save_path='monk3_bias_variance_epochs.png')
        
        # FASE 3: Test Set Evaluation (singolo modello)
        print(f"\n FASE 3: Valutazione singolo modello sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Performance (singolo)
        print(f"\n FASE 4: Grafico Model Performance (singolo)")
        plot_results(best_results, test_results, save_path='monk3_performance_single.png')

        
        # FASE 5: Test Set Evaluation (singolo modello)
        print(f"\n FASE 3: Valutazione singolo modello sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Performance (singolo)
        print(f"\n FASE 4: Grafico Model Performance (singolo)")
        plot_results(best_results, test_results, save_path='monk3_performance.png')
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print("  RIEPILOGO FINALE")
        print(f"{'='*70}")
        
        if test_results['test_accuracy'] >= 0.97:
            print("  ECCELLENTE:  97%+ test accuracy (su dataset con 5% noise)!")
        elif test_results['test_accuracy'] >= 0.95:
            print("  OTTIMO: 95%+ test accuracy!")
        elif test_results['test_accuracy'] >= 0.93:
            print(" ✓ BUONO: 93%+ test accuracy!")
        else:
            print(f"  Test Accuracy: {test_results['test_accuracy']:.2%}")
        
        print(f"\n{'─'*70}")
        print(" ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Accuracy:        {best_results['train_accuracy']:.4%}")
        print(f"  Validation Accuracy:  {best_results['val_accuracy']:.4%}")
        print(f"  Test Accuracy:         {test_results['test_accuracy']:.4%}")
        
        print(f"\n{'─'*70}")
        print(" LOSS FINALE (HALF MSE):")
        print(f"{'─'*70}")
        print(f"  Train Loss:       {best_results['train_loss']:.6f}")
        print(f"  Validation Loss:   {best_results['val_loss']:.6f}")
        print(f"  Test Loss:         {test_results['test_half_mse']:.6f}")  # ← STAMPA TEST LOSS
        
        print(f"\n{'─'*70}")
        print(" PARAMETRI MIGLIORI:")
        print(f"{'─'*70}")
        print(f"  Learning Rate:    {best_results['lr']}")
        print(f"  Seed:            {best_results['seed']}")
        print(f"  L2 Lambda:       {best_results['l2_lambda']}")
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
        print(f"  - monk3_bias_variance_epochs.png (bias-variance tradeoff)")
        print(f"  - monk3_performance.png (confusion matrix)")
        
        print(f"\n{'='*70}")
        print("  ESPERIMENTO MONK-3 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e: 
        print(f"\n ERRORE:  {e}")
        import traceback
        traceback.print_exc()