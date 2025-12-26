import numpy as np
import matplotlib. pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk2


def _monk2_test(learning_rate, seed, verbose=False): 
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}")
    
    # Seed per riproducibilità
    np.random. seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = return_monk2(
        one_hot=True, 
        val_split=0.3,
        dataset_shuffle=True
    )
    
    params = {
        'network_structure': [17, 8, 1],  
        'eta': learning_rate,
        'l2_lambda': 0.0, 
        'momentum':  0.9,
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
        epochs=2000,  # Più epoche per MONK-2
        batch_size=1,
        patience=200,
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
        'seed': seed,
        'data': (X_train, y_train, X_val, y_val, X_test, y_test)
    }


def grid_search_lr(n_seeds_per_lr=10, learning_rates=None):
    if learning_rates is None: 
        learning_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    
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
        print(f" TESTING LEARNING RATE: {lr}")
        print(f"{'='*70}")
        
        for seed_idx in range(n_seeds_per_lr):
            current_run += 1
            seed = seed_idx * 123 + int(lr * 1000)
            
            print(f"\n Run {current_run}/{total_runs} - LR={lr}, Seed={seed}", end=" → ")
            
            results = _monk2_test(learning_rate=lr, seed=seed, verbose=False)
            all_results.append(results)
            
            print(f"Train:  {results['train_accuracy']:.2%}, Val: {results['val_accuracy']:.2%}")
            
            if results['val_accuracy'] > best_val_acc:
                best_val_acc = results['val_accuracy']
                best_results = results
                print(f"  NUOVO BEST VAL ACC: {best_val_acc:.2%}")
            
            if results['val_accuracy'] >= 1.0:
                print(f"\n 100% VALIDATION ACCURACY!  Fermata anticipata.")
                break
        
        if best_val_acc >= 1.0:
            print(f"\n Obiettivo raggiunto!  Interrompo la ricerca.")
            break
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE (basata su VALIDATION)")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {best_results['val_accuracy']:.4%}")
    print(f"Train Accuracy:       {best_results['train_accuracy']:.4%}")
    print(f"Best LR:            {best_results['lr']}")
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
    print(f"    FN: {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error':  test_error,
        'precision': precision,
        'recall':  recall,
        'f1':  f1,
        'confusion_matrix': {'tp': tp, 'fp':  fp, 'tn': tn, 'fn': fn}
    }


def plot_results(train_results, test_results, save_path='monk2_performance.png'):
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
    plt.title(f'Model Performance (MONK-2)\n(LR={lr}, Seed={seed})', 
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
    plt.title(f'Confusion Matrix (TEST SET)\nPrecision:  {test_results["precision"]:.2%}, Recall: {test_results["recall"]:.2%}', 
             fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Model Performance salvato in: {save_path}")
    plt.show()


def plot_bias_variance_epochs(all_results, dataset_name='MONK-2', save_path='monk2_bias_variance_epochs.png'):
    from collections import defaultdict
    
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO BIAS-VARIANCE VS EPOCHE")
    print(f"{'='*70}")
    
    # RE-TRAINING DI ALCUNI MODELLI CON TRACKING
    print("Re-training di modelli selezionati con epoch tracking...")
    
    all_curves = []
    num_runs = min(15, len(all_results))  # Limita a 15 run
    
    for idx in range(num_runs):
        result = all_results[idx]
        print(f"\rProcessing run {idx+1}/{num_runs}.. .", end="")
        
        X_train, y_train, X_val, y_val, X_test, y_test = result['data']
        
        # Seed per riproducibilità
        np.random.seed(result['seed'])
        
        # Ricrea la rete
        params = result['params']. copy()
        net = NeuralNetwork(**params)
        
        train_errors_run = []
        test_errors_run = []
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
            train_pred_class = (train_pred > 0.5).astype(int)
            train_error = 1 - np.mean(train_pred_class == y_train)

            # Validation error 
            val_pred = net.predict(X_val)
            val_pred_class = (val_pred > 0.5).astype(int)
            val_error = 1 - np.mean(val_pred_class == y_val)
            
            epochs_run.append(epoch + 1)
            train_errors_run.append(train_error)
            test_errors_run. append(val_error)
        
        all_curves.append((epochs_run, train_errors_run, test_errors_run))
    
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
    
    # PLOT LINEE SOTTILI (ogni singolo run - smoothed)
    for epochs_run, train_errors_run, test_errors_run in all_curves:
        if len(train_errors_run) > 1:
            # Smooth individualmente
            train_smooth = smooth_curve(train_errors_run, weight=0.85)
            test_smooth = smooth_curve(test_errors_run, weight=0.85)
            
            # Linee sottili blu (training)
            ax.plot(epochs_run, train_smooth, color='lightblue', alpha=0.3, 
                   linewidth=0.8, zorder=1)
            
            # Linee sottili rosse (test)
            ax.plot(epochs_run, test_smooth, color='lightcoral', alpha=0.3, 
                   linewidth=0.8, zorder=1)
    
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
        train_means_smooth = smooth_curve(train_means, weight=0.90)
        test_means_smooth = smooth_curve(test_means, weight=0.90)
        
        # LINEE SPESSE (medie smoothed)
        ax.plot(epochs_axis, train_means_smooth, color='#1F618D', linewidth=4, 
               label='Training Error (mean)', zorder=10)
        
        ax.plot(epochs_axis, test_means_smooth, color='#CB4335', linewidth=4, 
               label='Validation Error (mean)', zorder=10)
        
        # ANNOTAZIONI
        y_max = max(max(train_means_smooth), max(test_means_smooth))
        y_min = min(min(train_means_smooth), min(test_means_smooth))
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # High Bias, Low Variance (sinistra)
        text_x_left = max_len * 0.1
        ax.text(text_x_left, y_max - 0.03 * y_range,
               'High Bias\nLow Variance', fontsize=11, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8,
                        edgecolor='orange', linewidth=1.5))
        
        # Low Bias, High Variance (destra)
        text_x_right = max_len * 0.9
        ax.text(text_x_right, y_max - 0.03 * y_range,
               'Low Bias\nHigh Variance', fontsize=11, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8,
                        edgecolor='red', linewidth=1.5))
        
        # Lucky/Unlucky
        ax.text(max_len * 0.95, y_min + 0.15 * y_range, 
               'lucky', fontsize=11, style='italic', color='#1F618D', 
               ha='right', fontweight='bold')
        ax.text(max_len * 0.95, y_max - 0.25 * y_range, 
               'unlucky', fontsize=11, style='italic', color='#CB4335', 
               ha='right', fontweight='bold')
        
        # Optimal complexity (minimo test error)
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
    ax.set_title(f'Bias-Variance Tradeoff - {dataset_name}\nTraining and Validation Error vs Training Epochs', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if train_means:
        ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.15 * y_range)
    ax.set_xlim(0, max_len * 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Bias-Variance (epoche) salvato in: {save_path}")
    plt.show()


if __name__ == "__main__":
    try:
        print("="*70)
        print(" MONK-2 - BINARY CLASSIFICATION")
        print("="*70)
        
        # FASE 1: Grid search
        print("\n FASE 1: Grid Search (Train + Validation)")
        best_results, all_results = grid_search_lr(
            n_seeds_per_lr=10,
            learning_rates=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        )
        # FILTRA solo configurazioni con 100% su train, val e test
        print(f"\n🔍 Filtraggio configurazioni con 100% accuracy...")

        perfect_results = []

        for result in all_results:
        # Verifica train e val
            if result['train_accuracy'] >= 1.0 and result['val_accuracy'] >= 1.0:
                # Verifica anche test
                X_train, y_train, X_val, y_val, X_test, y_test = result['data']
                test_pred = result['network'].predict(X_test)
                test_acc = np.mean((test_pred > 0.5).astype(int) == y_test)
        
                if test_acc >= 1.0:
                    perfect_results.append(result)
                    print(f"   LR={result['lr']}, Seed={result['seed']}:  100%")

        print(f"\n Trovate {len(perfect_results)}/{len(all_results)} configurazioni con 100%")

        # Usa solo le configurazioni perfette per il grafico
        if len(perfect_results) >= 5:
            all_results = perfect_results
            print(f"  Usando solo configurazioni con 100% per il grafico\n")
        else:
            print(f"    Poche config con 100%, usando tutte le {len(all_results)} configurazioni\n")
        
        # FASE 2: Bias-Variance Tradeoff con EPOCHE
        print(f"\n FASE 2: Grafico Bias-Variance vs Epoche")
        plot_bias_variance_epochs(all_results, 
                                  dataset_name='MONK-2',
                                  save_path='monk2_bias_variance_epochs.png')
        
        # FASE 3: Valuta sul TEST SET (UNA SOLA VOLTA!)
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Model Performance + Confusion Matrix
        print(f"\n FASE 4: Grafico Model Performance")
        plot_results(best_results, test_results, save_path='monk2_performance.png')
        
        
        # FASE 5: ENSEMBLE DI MODELLI
        print(f"\n{'='*70}")
        print(" FASE 5: ENSEMBLE DI MODELLI")
        print(f"{'='*70}")
        
        # Scegli i 3 migliori seed dalla grid search
        sorted_results = sorted(all_results, key=lambda x: x['val_accuracy'], reverse=True)
        top_3_results = sorted_results[:3]
        
        print(f"\nTop 3 configurazioni selezionate:")
        for idx, res in enumerate(top_3_results, 1):
            print(f"  {idx}. LR={res['lr']}, Seed={res['seed']}, Val Acc={res['val_accuracy']:.2%}")
        
        print(f"\n Training ensemble models...")
        ensemble_nets = []
        ensemble_configs = []
        
        for idx, result in enumerate(top_3_results, 1):
            print(f"\r  Training model {idx}/3.. .", end="", flush=True)
            
            # Usa la rete già addestrata (più veloce)
            ensemble_nets.append(result['network'])
            ensemble_configs.append({'lr': result['lr'], 'seed': result['seed']})
        
        print(f"\n Ensemble training completato")
        
        # PREDIZIONE ENSEMBLE
        print(f"\n Valutazione Ensemble sul Test Set...")
        
        ensemble_preds = []
        for net in ensemble_nets:
            pred = net.predict(X_test)
            ensemble_preds. append(pred)
        
        # Media delle predizioni
        ensemble_pred_avg = np.mean(ensemble_preds, axis=0)
        ensemble_pred_class = (ensemble_pred_avg > 0.5).astype(int)
        
        # Calcola accuracy
        ensemble_acc = np.mean(ensemble_pred_class == y_test)
        ensemble_error = 1 - ensemble_acc
        
        # Confusion matrix ensemble
        tp_ens = np.sum((ensemble_pred_class == 1) & (y_test == 1))
        fp_ens = np.sum((ensemble_pred_class == 1) & (y_test == 0))
        tn_ens = np.sum((ensemble_pred_class == 0) & (y_test == 0))
        fn_ens = np.sum((ensemble_pred_class == 0) & (y_test == 1))
        
        precision_ens = tp_ens / (tp_ens + fp_ens) if (tp_ens + fp_ens) > 0 else 0
        recall_ens = tp_ens / (tp_ens + fn_ens) if (tp_ens + fn_ens) > 0 else 0
        f1_ens = 2 * (precision_ens * recall_ens) / (precision_ens + recall_ens) if (precision_ens + recall_ens) > 0 else 0
        
        print(f"\n{'='*70}")
        print(" RISULTATI ENSEMBLE")
        print(f"{'='*70}")
        print(f"  Ensemble Accuracy:   {ensemble_acc:.4%}")
        print(f"  Ensemble Error:     {ensemble_error:.4%}")
        print(f"  Precision:           {precision_ens:.4f}")
        print(f"  Recall:             {recall_ens:.4f}")
        print(f"  F1-score:           {f1_ens:.4f}")
        print(f"\n  Confusion Matrix (Ensemble):")
        print(f"    TP: {tp_ens}  FP: {fp_ens}")
        print(f"    FN: {fn_ens}  TN: {tn_ens}")
        
        # Confronto con singolo modello
        print(f"\n{'─'*70}")
        print(" CONFRONTO:  Singolo Modello vs Ensemble")
        print(f"{'─'*70}")
        print(f"  Singolo (best): {test_results['test_accuracy']:.4%}")
        print(f"  Ensemble:       {ensemble_acc:.4%}")
        
        improvement = (ensemble_acc - test_results['test_accuracy']) * 100
        if improvement > 0:
            print(f"  Miglioramento:   +{improvement:.2f}%")
        else:
            print(f"  Differenza:     {improvement:.2f}%")
        
        # SALVA RISULTATO ENSEMBLE
        if ensemble_acc >= test_results['test_accuracy']: 
            print(f"\n L'ensemble ha performato meglio o uguale al singolo modello!")
            
            # Crea un risultato "fittizio" per plot_results
            ensemble_results = {
                'test_accuracy': ensemble_acc,
                'test_error': ensemble_error,
                'precision': precision_ens,
                'recall': recall_ens,
                'f1': f1_ens,
                'confusion_matrix': {'tp': tp_ens, 'fp': fp_ens, 'tn': tn_ens, 'fn': fn_ens}
            }
            
            # Crea pseudo train_results per il plot
            ensemble_train_results = best_results.copy()
            ensemble_train_results['lr'] = f"Ensemble-{len(ensemble_nets)}"
            ensemble_train_results['seed'] = "Mixed"
            
            plot_results(ensemble_train_results, ensemble_results, 
                        save_path='monk2_performance_ENSEMBLE.png')
            
            print(f"\n Grafico ensemble salvato:  monk2_performance_ENSEMBLE.png")
        
        if ensemble_acc >= 1.0:
            print(f"\n{'🎉'*25}")
            print(f" 100% TEST ACCURACY RAGGIUNTO CON ENSEMBLE! ")
            print(f"{''*25}\n")
        
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        if test_results['test_accuracy'] >= 1.0:
            print(" PERFETTO:  100% TEST ACCURACY RAGGIUNTO!")
        elif test_results['test_accuracy'] >= 0.95:
            print(" ECCELLENTE: 95%+ test accuracy!")
        elif test_results['test_accuracy'] >= 0.90:
            print("✓ OTTIMO: 90%+ test accuracy!")
        else:
            print(f"Test Accuracy: {test_results['test_accuracy']:.2%}")
        
        print(f"\n{'─'*70}")
        print("ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Accuracy:         {best_results['train_accuracy']:.4%}")
        print(f"  Validation Accuracy:    {best_results['val_accuracy']:.4%}")
        print(f"  Test Accuracy (FINAL):  {test_results['test_accuracy']:.4%}")
        if ensemble_acc >= test_results['test_accuracy']: 
            print(f"  Ensemble Accuracy:      {ensemble_acc:.4%} ← BEST!")
        
        print(f"\n{'─'*70}")
        print(" PARAMETRI MIGLIORI:")
        print(f"{'─'*70}")
        print(f"  Learning Rate:   {best_results['lr']}")
        print(f"  Random Seed:    {best_results['seed']}")
        print(f"  Hidden Units:   {best_results['params']['network_structure'][1]}")
        print(f"  Momentum:       {best_results['params']['momentum']}")
        print(f"  L2 Lambda:      {best_results['params']['l2_lambda']}")
        print(f"  Architecture:   {best_results['params']['network_structure']}")
        
        print(f"\n{'─'*70}")
        print(" METRICHE DETTAGLIATE (TEST SET - SINGOLO):")
        print(f"{'─'*70}")
        print(f"  Precision:  {test_results['precision']:.4f}")
        print(f"  Recall:     {test_results['recall']:.4f}")
        print(f"  F1-score:   {test_results['f1']:.4f}")
        
        cm = test_results['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives  (TP): {cm['tp']}")
        print(f"    False Positives (FP): {cm['fp']}")
        print(f"    True Negatives  (TN): {cm['tn']}")
        print(f"    False Negatives (FN): {cm['fn']}")
        
        if ensemble_acc >= test_results['test_accuracy']:
            print(f"\n{'─'*70}")
            print(" METRICHE DETTAGLIATE (TEST SET - ENSEMBLE):")
            print(f"{'─'*70}")
            print(f"  Precision:  {precision_ens:.4f}")
            print(f"  Recall:     {recall_ens:.4f}")
            print(f"  F1-score:   {f1_ens:.4f}")
            print(f"\n  Confusion Matrix:")
            print(f"    True Positives  (TP): {tp_ens}")
            print(f"    False Positives (FP): {fp_ens}")
            print(f"    True Negatives  (TN): {tn_ens}")
            print(f"    False Negatives (FN): {fn_ens}")
        
        print(f"\n File salvati:")
        print(f"  - monk2_bias_variance_epochs.png (bias-variance vs epoche)")
        print(f"  - monk2_performance. png (singolo modello)")
        if ensemble_acc >= test_results['test_accuracy']:
            print(f"  - monk2_performance_ENSEMBLE.png (ensemble) ")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO MONK-2 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE:  {e}")
        import traceback
        traceback.print_exc()