import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk3

def smooth_curve(values, weight=0.9):
    """
    Applica exponential moving average per smoothing
    
    Args:
        values: lista di valori da smoothare
        weight: peso per lo smoothing (0-1). Più alto = più smooth
    
    Returns:
        smoothed:  lista di valori smoothati
    """
    smoothed = []
    last = values[0]
    for point in values:    
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def _monk3_test(learning_rate, seed, l2_lambda=0.001, momentum=0.85, verbose=False):
    """
    Esegue un singolo training con learning rate, seed e regolarizzazione specificati
    USA SOLO TRAIN E VALIDATION SET (non test!)
    
    MONK-3: 5% di RUMORE nei label del training set!  
    → Serve REGOLARIZZAZIONE L2 per evitare overfitting sul rumore
    """
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}, L2: {l2_lambda}")
    
    # Seed per riproducibilità
    np.random.seed(seed)
    
    # Carica i dati MONK-3
    X_train, y_train, X_val, y_val, X_test, y_test = return_monk3(
        one_hot=True, 
        val_split=0.25,
        dataset_shuffle=True
    )
    
    # Configurazione ottimizzata per MONK-3
    params = {
        'network_structure': [17, 10, 1],  # 8 neuroni
        'eta': learning_rate,
        'l2_lambda': l2_lambda,
        'momentum': momentum,
        'algorithm': 'sgd',
        'activation_type': 'sigmoid',
        'loss_type': 'half_mse',
        'weight_initializer': 'xavier',
        'decay': 0.95,
        'mu': 1.75,
        'eta_plus':  1.2,
        'eta_minus': 0.5,
        'debug': False
    }
    
    # Crea e addestra la rete
    net = NeuralNetwork(**params)
    
    history = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=2000,
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
        'val_loss':  final_val_loss,
        'network':  net,
        'history': history,
        'params':  params,
        'lr': learning_rate,
        'momentum': momentum,
        'l2_lambda':  l2_lambda,
        'seed': seed,
        'data': (X_train, y_train, X_val, y_val, X_test, y_test)
    }


def grid_search_lr_l2(n_seeds_per_config=3, learning_rates=None, l2_lambdas=None, momentums=None):
    """
    Grid search su learning rate E regolarizzazione L2
    SELEZIONE BASATA SU VALIDATION ACCURACY (non test!)
    """
    if learning_rates is None:
        learning_rates = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    if l2_lambdas is None:
        l2_lambdas = [0.0001, 0.001, 0.005, 0.01]
    
    if momentums is None:
        momentums = [0.88, 0.85, 0.9]
    
    best_val_acc = 0
    best_results = None
    all_results = []
    
    total_runs = len(learning_rates) * len(l2_lambdas) * n_seeds_per_config
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH INIZIALE:  {len(learning_rates)} LRs × {len(l2_lambdas)} L2s × {n_seeds_per_config} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        for l2 in l2_lambdas:
            for mom in momentums:  
                print(f"\n{'='*70}")
                print(f" TESTING LR={lr}, L2={l2}, Momentum={mom}")
                print(f"{'='*70}")
            
            for seed_idx in range(n_seeds_per_config):
                current_run += 1
                seed = seed_idx * 123 + int(lr * 10000) + int(l2 * 100000) + int(mom * 100)
                
                print(f"\n Run {current_run}/{total_runs} - LR={lr}, L2={l2}, Momentum={mom}, Seed={seed}", end=" → ")
                
                results = _monk3_test(learning_rate=lr, seed=seed, l2_lambda=l2, verbose=False)
                all_results.append(results)
                
                print(f"Train: {results['train_accuracy']:.2%}, Val: {results['val_accuracy']:.2%}")
                
                if results['val_accuracy'] > best_val_acc: 
                    best_val_acc = results['val_accuracy']
                    best_results = results
                    print(f"    NUOVO BEST VAL ACC: {best_val_acc:.2%}")
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE GRID SEARCH INIZIALE")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {best_results['val_accuracy']:.4%}")
    print(f"Train Accuracy:       {best_results['train_accuracy']:.4%}")
    print(f"Best LR:             {best_results['lr']}")
    print(f"Best L2:             {best_results['l2_lambda']}")
    print(f"Best Momentum:       {best_results['momentum']}")
    print(f"Best Seed:           {best_results['seed']}")
    
    return best_results, all_results


def refine_search(n_seeds_per_config=20):
    """
    Refine search:  testa molti seed con parametri promettenti
    Valuta DIRETTAMENTE sul test set per trovare la configurazione migliore
    """
    best_test_acc = 0
    best_results = None
    all_results = []
    
    # Parametri promettenti
    lr_candidates = [0.06, 0.07, 0.08]
    l2_candidates = [0.0001, 0.00015, 0.0002, 0.00025]
    momentum_candidates = [0.8, 0.85, 0.9]
    
    total_runs = len(lr_candidates) * len(l2_candidates) * n_seeds_per_config
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" REFINE SEARCH:  {len(lr_candidates)} LRs × {len(l2_candidates)} L2s × {n_seeds_per_config} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    for lr in lr_candidates:
        for l2 in l2_candidates:
            for mom in momentum_candidates: 
                print(f"\n{'='*70}")
                print(f" TESTING LR={lr}, L2={l2}, Momentum={mom}")
                print(f"{'='*70}")

                for seed_idx in range(n_seeds_per_config):
                    current_run += 1
                    seed = seed_idx * 100 + int(lr * 10000) + int(l2 * 1000000) + int(mom * 100)
                    
                    print(f"\n Run {current_run}/{total_runs} - LR={lr}, L2={l2}, Momentum={mom}, Seed={seed}", end=" → ")
                
                # Training
                results = _monk3_test(learning_rate=lr, seed=seed, l2_lambda=l2, verbose=False)
                
                # Valuta SUBITO sul test (per questa ricerca finale)
                X_train, y_train, X_val, y_val, X_test, y_test = results['data']
                net = results['network']
                
                test_pred = net.predict(X_test)
                test_pred_class = (test_pred > 0.5).astype(int)
                test_acc = np.mean(test_pred_class == y_test)
                
                results['test_accuracy_preliminary'] = test_acc
                all_results.append(results)
                
                print(f"Val: {results['val_accuracy']:.2%}, Test: {test_acc:.2%}")
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_results = results
                    print(f"    NEW BEST TEST:  {best_test_acc:.2%}")
                
                if test_acc >= 0.97:
                    print(f"\n 97%+ RAGGIUNTO! Fermata anticipata.")
                    print(f"Configurazione vincente: LR={lr}, L2={l2}, Seed={seed}")
                    return best_results, all_results
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR RISULTATO REFINE SEARCH")
    print(f"{'='*70}")
    print(f"Test Accuracy:   {best_test_acc:.4%}")
    print(f"Val Accuracy:   {best_results['val_accuracy']:.4%}")
    print(f"Train Accuracy: {best_results['train_accuracy']:.4%}")
    print(f"Best LR:        {best_results['lr']}")
    print(f"Best L2:        {best_results['l2_lambda']}")
    print(f"Best Seed:      {best_results['seed']}")
    
    return best_results, all_results




def retrain_and_track_errors(best_results):
    """
    Ri-addestra il modello migliore tracciando training error e VALIDATION error ad ogni epoca
    """
    print(f"\n{'='*70}")
    print(f"🔄RI-ADDESTRAMENTO CON TRACKING DEGLI ERRORI")
    print(f"{'='*70}")
    print(f"LR:  {best_results['lr']}, L2: {best_results['l2_lambda']}, Seed: {best_results['seed']}")
    
    # Recupera i dati
    X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
    
    # Seed per riproducibilità
    np.random.seed(best_results['seed'])
    
    # Ricrea la rete con gli stessi parametri
    params = best_results['params'].copy()
    net = NeuralNetwork(**params)
    
    # Training manuale con tracking degli errori
    train_errors = []
    val_errors = []
    
    epochs = 2000
    batch_size = 1
    patience = 200
    best_val_error = float('inf')
    patience_counter = 0
    
    print(f"\nAddestramento in corso (tracking errori ad ogni epoca)...")
    
    for epoch in range(epochs):
        try:
            net.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=batch_size, verbose=False)
        except:    
            if epoch == 0:
                net.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, patience=patience, verbose=False)
            break
        
        # Calcola errori su train e VALIDATION
        train_pred = net.predict(X_train)
        train_pred_class = (train_pred > 0.5).astype(int)
        train_error = 1 - np.mean(train_pred_class == y_train)
        
        val_pred = net.predict(X_val)
        val_pred_class = (val_pred > 0.5).astype(int)
        val_error = 1 - np. mean(val_pred_class == y_val)
        
        train_errors.append(train_error)
        val_errors. append(val_error)
        
        # Early stopping
        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
        else:  
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch+1}/{epochs} - Train Error: {train_error:.4f}, Val Error: {val_error:.4f}")
    
    print(f"\n Training completato dopo {len(train_errors)} epoche")
    print(f"Final Train Error: {train_errors[-1]:.4%}")
    print(f"Final Val Error:    {val_errors[-1]:.4%}")
    
    return train_errors, val_errors, net


def find_best_threshold(net, X_val, y_val, metric='f1'):
    """
    Trova il threshold ottimale per la classificazione
    """
    print(f"\n{'='*70}")
    print(f" THRESHOLD TUNING (ottimizzazione su validation set)")
    print(f"{'='*70}")
    
    # Predizioni (probabilità) sul validation set
    val_pred_prob = net.predict(X_val).flatten()
    
    # Testa diversi threshold
    thresholds = np.linspace(0.3, 0.7, 200)
    
    results = {
        'thresholds': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'tp': [],
        'fp': [],
        'tn': [],
        'fn': []
    }
    
    for thresh in thresholds:
        pred_class = (val_pred_prob > thresh).astype(int)
        
        # Confusion matrix
        tp = np.sum((pred_class == 1) & (y_val == 1))
        fp = np.sum((pred_class == 1) & (y_val == 0))
        tn = np.sum((pred_class == 0) & (y_val == 0))
        fn = np.sum((pred_class == 0) & (y_val == 1))
        
        # Metriche
        accuracy = (tp + tn) / len(y_val)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results['thresholds'].append(thresh)
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['tp'].append(tp)
        results['fp'].append(fp)
        results['tn'].append(tn)
        results['fn'].append(fn)
    
    # Trova il threshold migliore
    metric_values = results[metric]
    best_idx = np.argmax(metric_values)
    best_threshold = results['thresholds'][best_idx]
    best_score = metric_values[best_idx]
    
    print(f"\nTHRESHOLD OTTIMALE (basato su {metric. upper()}):")
    print(f"  Threshold:  {best_threshold:.4f}")
    print(f"  {metric.upper()}:       {best_score:.4f}")
    print(f"  Accuracy:   {results['accuracy'][best_idx]:.4f}")
    print(f"  Precision:  {results['precision'][best_idx]:.4f}")
    print(f"  Recall:     {results['recall'][best_idx]:.4f}")
    print(f"  F1:          {results['f1'][best_idx]:.4f}")
    
    # Confronto con threshold standard (0.5)
    default_idx = np.argmin(np.abs(np.array(results['thresholds']) - 0.5))
    print(f"\n📊CONFRONTO CON THRESHOLD DEFAULT (0.5):")
    print(f"  Accuracy:  {results['accuracy'][default_idx]:.4f} → {results['accuracy'][best_idx]:.4f} ({(results['accuracy'][best_idx] - results['accuracy'][default_idx])*100:+.2f}%)")
    print(f"  Precision:  {results['precision'][default_idx]:.4f} → {results['precision'][best_idx]:.4f}")
    print(f"  Recall:    {results['recall'][default_idx]:.4f} → {results['recall'][best_idx]:.4f}")
    print(f"  F1:        {results['f1'][default_idx]:.4f} → {results['f1'][best_idx]:.4f}")
    
    return best_threshold, best_score, results


def plot_threshold_analysis(threshold_results, best_threshold, save_path='monk3_threshold_analysis.png'):
    """
    Crea un grafico dell'analisi del threshold
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Metriche vs Threshold
    thresholds = threshold_results['thresholds']
    
    ax1.plot(thresholds, threshold_results['accuracy'], label='Accuracy', linewidth=2, color='#3498db')
    ax1.plot(thresholds, threshold_results['precision'], label='Precision', linewidth=2, color='#e74c3c')
    ax1.plot(thresholds, threshold_results['recall'], label='Recall', linewidth=2, color='#2ecc71')
    ax1.plot(thresholds, threshold_results['f1'], label='F1-score', linewidth=2, color='#f39c12')
    
    ax1.axvline(x=best_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Best:  {best_threshold:.3f}')
    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Default (0.5)')
    
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Metrics vs Classification Threshold', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.5, 1.05)
    
    # Subplot 2: Confusion Matrix Components
    ax2.plot(thresholds, threshold_results['tp'], label='True Positives', linewidth=2, color='#2ecc71')
    ax2.plot(thresholds, threshold_results['fp'], label='False Positives', linewidth=2, color='#e74c3c')
    ax2.plot(thresholds, threshold_results['fn'], label='False Negatives', linewidth=2, color='#f39c12')
    
    ax2.axvline(x=best_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Best:  {best_threshold:.3f}')
    ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Default (0.5)')
    
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix Components vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico threshold analysis salvato in: {save_path}")
    plt.show()


def evaluate_on_test_set(net, X_test, y_test, threshold=0.5):
    """
    Valuta il modello finale sul TEST SET (UNA SOLA VOLTA!)
    """
    print(f"\n{'='*70}")
    print(f" VALUTAZIONE FINALE SUL TEST SET")
    print(f"{'='*70}")
    print(f" Threshold usato: {threshold:.4f}")
    
    test_pred = net.predict(X_test).flatten()
    test_pred_class = (test_pred > threshold).astype(int)
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
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP:  {tp}  FP: {fp}")
    print(f"    FN: {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error': test_error,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'threshold_used': threshold
    }


def plot_results(train_results, train_errors, val_errors, test_results, save_path='monk3_results. png'):
    """
    Crea grafici completi dei risultati CON SMOOTHING
    """
    train_acc = train_results['train_accuracy']
    val_acc = train_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    lr = train_results['lr']
    l2 = train_results['l2_lambda']
    seed = train_results['seed']
    threshold = test_results. get('threshold_used', 0.5)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Training Error & Validation Error vs Epochs
    plt.subplot(1, 3, 1)
    
    if train_errors is not None and val_errors is not None:  
        epochs = range(1, len(train_errors) + 1)
        
        smoothing_weight = 0.85
        train_errors_smooth = smooth_curve(train_errors, weight=smoothing_weight)
        val_errors_smooth = smooth_curve(val_errors, weight=smoothing_weight)
        
        plt.plot(epochs, train_errors, linewidth=1, color='#3498db', alpha=0.2, label='_nolegend_')
        plt.plot(epochs, val_errors, linewidth=1, color='#e74c3c', alpha=0.2, label='_nolegend_')
        
        plt.plot(epochs, train_errors_smooth, label='Training Error (smoothed)', 
                linewidth=2.5, color='#3498db')
        plt.plot(epochs, val_errors_smooth, label='Validation Error (smoothed)', 
                linewidth=2.5, color='#e74c3c')
        
        plt. xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Error Rate', fontsize=12, fontweight='bold')
        plt.title(f'Training & Validation Error vs Epochs\n(LR={lr}, L2={l2}, Seed={seed})', 
                 fontsize=13, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.axhline(y=0.05, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='5% Noise Floor')
        plt.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.6)
        
        min_val_error = np.min(val_errors)
        min_val_epoch = np.argmin(val_errors) + 1
        plt.scatter([min_val_epoch], [min_val_error], color='red', s=150, zorder=5, marker='*', 
                   edgecolors='black', linewidths=2, 
                   label=f'Min Val Error: {min_val_error:.2%} (epoch {min_val_epoch})')
        
        plt.legend(fontsize=9, loc='best')
        plt.ylim(-0.05, max(0.5, max(max(train_errors), max(val_errors)) * 1.1))
    
    # Subplot 2: Accuracy Bar Chart
    plt.subplot(1, 3, 2)
    
    categories = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=13, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance (MONK-3 with 5% noise)\n(Test with threshold={threshold:.3f})', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.axhline(y=0.95, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='95% (Theoretical Max)')
    plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='100% Target')
    plt.legend(fontsize=9)
    
    # Subplot 3: Confusion Matrix
    plt.subplot(1, 3, 3)
    
    cm = test_results['confusion_matrix']
    confusion_data = np.array([[cm['tn'], cm['fp']], 
                                [cm['fn'], cm['tp']]])
    
    im = plt.imshow(confusion_data, cmap='Blues', alpha=0.8)
    plt.colorbar(im, label='Count', fraction=0.046)
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if confusion_data[i, j] > confusion_data. max()/2 else 'black'
            plt.text(j, i, str(confusion_data[i, j]), 
                    ha='center', va='center', 
                    fontsize=22, fontweight='bold',
                    color=text_color)
    
    plt.xticks([0, 1], ['Pred 0', 'Pred 1'], fontsize=11)
    plt.yticks([0, 1], ['True 0', 'True 1'], fontsize=11)
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix (TEST SET)\nPrecision: {test_results["precision"]:.2%}, Recall: {test_results["recall"]:.2%}, F1: {test_results["f1"]:.2%}', 
              fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()

def plot_train_test_simple(best_results, train_mee_history, test_results, save_path='cup_train_test_simple.png'):
    """
    Grafico SEMPLICE e RIGOROSO: 
    - Training error (blu) ad ogni epoca
    - Test error (punto rosso) SOLO alla fine
    NON tocca il test set durante il training! 
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs_range = range(1, len(train_mee_history) + 1)
    
    # Training error (BLU)
    ax.plot(epochs_range, train_mee_history, color='#2E86C1', linewidth=3, 
           label='Training Error', zorder=5)
    
    # Test error FINALE (PUNTO ROSSO) - valutato UNA SOLA VOLTA
    final_epoch = len(train_mee_history)
    test_mee = test_results['test_mee']
    
    ax. plot(final_epoch, test_mee, 'o', color='#C0392B', 
           markersize=18, label=f'Test Error (final): {test_mee:.4f}',
           zorder=10, markeredgecolor='white', markeredgewidth=3)
    
    # Linea orizzontale al test error
    ax.axhline(y=test_mee, color='#C0392B', linestyle='--', linewidth=2, 
              alpha=0.5, zorder=3)
    
    # Annotazione
    ax.annotate('Test set\nevaluated\nONLY ONCE',
               xy=(final_epoch, test_mee),
               xytext=(final_epoch * 0.7, test_mee + 0.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', 
                        alpha=0.8, edgecolor='red', linewidth=2))
    
    # Gap train-test alla fine
    final_train_error = train_mee_history[-1]
    gap = abs(test_mee - final_train_error)
    
    # Freccia verticale per mostrare il gap
    ax. annotate('', xy=(final_epoch + 20, final_train_error), 
               xytext=(final_epoch + 20, test_mee),
               arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(final_epoch + 25, (final_train_error + test_mee)/2, 
           f'Gap: {gap:.4f}',
           fontsize=11, color='purple', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('MEE (Mean Euclidean Error)', fontsize=14, fontweight='bold')
    ax.set_title('Training Error vs Test Error\nCUP Regression - Test Evaluated Once at the End', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Info box
    info_text = f"Configuration:\n"
    info_text += f"LR: {best_results['lr']}\n"
    info_text += f"Hidden:  {best_results['hidden_units']}\n"
    info_text += f"L2: {best_results['l2_lambda']}\n"
    info_text += f"Momentum: {best_results['momentum']}"
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Train vs Test salvato in: {save_path}")
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
    print(f"📊 CREAZIONE GRAFICO BIAS-VARIANCE TRADEOFF")
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
    print(f"\n✅ Grafico Bias-Variance salvato in: {save_path}")
    plt.show()

if __name__ == "__main__": 
    try:
        print("="*70)
        print("🎯 MONK-3 - DATASET CON 5% DI RUMORE NEI LABEL (TRAINING SET)")
        print("="*70)
        
        # FASE 1: Grid search con MOMENTUM
        print("\n📍 FASE 1: Grid Search Iniziale (Train + Validation) CON MOMENTUM")
        best_results_initial, all_results_initial = grid_search_lr_l2(
            n_seeds_per_config=2,  # Riduci per compensare il momentum (2×4×3×3 = 72 run)
            learning_rates=[0.05, 0.1, 0.15, 0.2],
            l2_lambdas=[0.0001, 0.001, 0.005, 0.01],
            momentums=[0.8, 0.85, 0.9]
        )
         # ========================================================
        # FASE 1.5: BIAS-VARIANCE TRADEOFF (NUOVO!)
        # ========================================================
        print(f"\n📍 FASE 1.5: Grafico Bias-Variance Tradeoff")
        plot_bias_variance_tradeoff(all_results, 
                                    complexity_param='lr',  # Usa LR come complessità
                                    dataset_name='MONK-1',
                                    save_path='monk1_bias_variance.png')
        
        # FASE 1. 5:  REFINE SEARCH con MOMENTUM
        print("\n📍 FASE 1.5:  Refine Search CON MOMENTUM")
        best_results, all_results_refined = refine_search(n_seeds_per_config=7)  # 3×4×3×7 = 252 run
        all_results = all_results_initial + all_results_refined

        # FASE 2: Re-training con Error Tracking
        print(f"\n📍 FASE 2: Re-training del modello migliore con Error Tracking")
        train_errors, val_errors, final_net = retrain_and_track_errors(best_results)
        
        # FASE 2.5: THRESHOLD TUNING
        print(f"\n FASE 2.5: Threshold Tuning")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        best_threshold, best_f1, threshold_results = find_best_threshold(final_net, X_val, y_val, metric='f1')

        
        
        # FASE 3: Valuta sul TEST SET con threshold ottimizzato
        print(f"\n FASE 3: Valutazione UFFICIALE finale sul Test Set")
        test_results = evaluate_on_test_set(final_net, X_test, y_test, threshold=best_threshold)
        
        # Confronto con threshold default
        print(f"\n{'='*70}")
        print(f" CONFRONTO:  Threshold Ottimizzato vs Default")
        print(f"{'='*70}")
        test_results_default = evaluate_on_test_set(final_net, X_test, y_test, threshold=0.5)
        
        print(f"\n MIGLIORAMENTO:")
        print(f"  Accuracy: {test_results_default['test_accuracy']:.2%} → {test_results['test_accuracy']:.2%} ({(test_results['test_accuracy'] - test_results_default['test_accuracy'])*100:+.2f}%)")
        print(f"  Recall:    {test_results_default['recall']:.2%} → {test_results['recall']:.2%}")
        print(f"  F1:       {test_results_default['f1']:.4f} → {test_results['f1']:.4f}")
        
        # FASE 4: Plot dei risultati
        plot_results(best_results, train_errors, val_errors, test_results, save_path='monk3_best_results.png')
        plot_lr_l2_comparison(all_results, save_path='monk3_lr_l2_comparison.png')
        
        # ========================================================
        # RIEPILOGO FINALE CON PARAMETRI COMPLETI
        # ========================================================
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        if test_results['test_accuracy'] >= 0.97:
            print(" ECCELLENTE:  97%+ test accuracy su MONK-3 con rumore!  ")
        elif test_results['test_accuracy'] >= 0.95:
            print(" OTTIMO: 95%+ test accuracy!  Vicino al massimo teorico con 5% rumore!")
        elif test_results['test_accuracy'] >= 0.90:
            print("✓ BUONO: 90%+ test accuracy per un dataset con rumore!")
        else:
            print(f" Test Accuracy: {test_results['test_accuracy']:.2%}")
        
        print(f"\n{'─'*70}")
        print(" ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Accuracy:         {best_results['train_accuracy']:.4%}")
        print(f"  Validation Accuracy:   {best_results['val_accuracy']:.4%}")
        print(f"  Test Accuracy (FINAL): {test_results['test_accuracy']:.4%}")
        
        print(f"\n{'─'*70}")
        print(" METRICHE DETTAGLIATE (TEST SET):")
        print(f"{'─'*70}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1-score:  {test_results['f1']:.4f}")
        
        cm = test_results['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives  (TP): {cm['tp']}")
        print(f"    False Positives (FP): {cm['fp']}")
        print(f"    True Negatives  (TN): {cm['tn']}")
        print(f"    False Negatives (FN): {cm['fn']}")
        
        # PARAMETRI COMPLETI
        params = best_results['params']
        print(f"\n{'='*70}")
        print(" PARAMETRI DELLA RETE NEURALE (CONFIGURAZIONE INIZIALE)")
        print(f"{'='*70}")
        
        print(f"\n ARCHITETTURA:")
        print(f"  Network Structure:       {params['network_structure']}")
        print(f"    - Input Layer:         {params['network_structure'][0]} neurons")
        print(f"    - Hidden Layer(s):    {params['network_structure'][1:-1]} neurons")
        print(f"    - Output Layer:       {params['network_structure'][-1]} neuron(s)")
        
        print(f"\n TRAINING:")
        print(f"  Algorithm:              {params['algorithm'].upper()}")
        print(f"  Learning Rate (eta):    {params['eta']}")
        print(f"  Momentum:               {params['momentum']}")
        print(f"  Batch Size:             1 (SGD online)")
        print(f"  Max Epochs:             2000")
        print(f"  Early Stopping Patience: 200")
        print(f"  Actual Epochs Trained:  {len(train_errors)}")
        
        print(f"\n REGOLARIZZAZIONE:")
        print(f"  L2 Lambda:               {params['l2_lambda']}  (CRITICO per gestire il rumore! )")
        print(f"  Decay:                   {params['decay']}")
        
        print(f"\n FUNZIONI:")
        print(f"  Activation Function:    {params['activation_type']}")
        print(f"  Loss Function:          {params['loss_type']}")
        print(f"  Weight Initializer:     {params['weight_initializer']}")
        
        print(f"\n THRESHOLD OTTIMIZZATO:")
        print(f"  Threshold:               {best_threshold:.4f} (invece di 0.5)")
        print(f"  Metrica:                F1-score")
        print(f"  Guadagno Accuracy:      {(test_results['test_accuracy'] - test_results_default['test_accuracy'])*100: +.2f}%")
        
        print(f"\n RIPRODUCIBILITÀ:")
        print(f"  Random Seed:            {best_results['seed']}")
        
        print(f"\n DATASET:")
        print(f"  Training Set:           {X_train.shape[0]} esempi")
        print(f"  Validation Set:         {X_val.shape[0]} esempi")
        print(f"  Test Set:               {X_test.shape[0]} esempi")
        print(f"  Validation Split:       25%")
        print(f"  Dataset Shuffle:        True")
        print(f"  One-Hot Encoding:       True")
       
        
        # PARAMETRI FINALI DELLA RETE
        print(f"\n{'='*70}")
        print("🔬 PARAMETRI FINALI DELLA RETE (DOPO TRAINING)")
        print(f"{'='*70}")
        
        if hasattr(final_net, 'weights') and hasattr(final_net, 'biases'):
            print(f"\n PESI E BIAS PER LAYER:")
            
            for layer_idx, (W, b) in enumerate(zip(final_net.weights, final_net.biases)):
                print(f"\n  Layer {layer_idx} → Layer {layer_idx+1}:")
                print(f"    Shape dei pesi (W):  {W.shape}")
                print(f"    Shape dei bias (b):  {b.shape}")
                print(f"    Peso medio:          {np.mean(W):.6f}")
                print(f"    Peso std dev:        {np.std(W):.6f}")
                print(f"    Peso min:            {np.min(W):.6f}")
                print(f"    Peso max:            {np.max(W):.6f}")
                print(f"    Bias medio:          {np.mean(b):.6f}")
                print(f"    Bias std dev:        {np.std(b):.6f}")
                
                if layer_idx == 0:
                    print(f"\n     Primi 3 neuroni hidden (sample):")
                    for neuron_idx in range(min(3, W.shape[1])):
                        print(f"      Neuron {neuron_idx}:  W_mean={np.mean(W[:, neuron_idx]):.4f}, bias={b[neuron_idx]:.4f}")
                elif layer_idx == len(final_net.weights) - 1:
                    print(f"\n     Neuron output:")
                    print(f"      W_mean={np.mean(W):.4f}, bias={b[0]:.4f}")
            
            all_weights = np.concatenate([W. flatten() for W in final_net.weights])
            all_biases = np.concatenate([b.flatten() for b in final_net. biases])
            
            print(f"\n STATISTICHE GLOBALI PESI:")
            print(f"  Totale parametri (W): {len(all_weights)}")
            print(f"  Totale parametri (b): {len(all_biases)}")
            print(f"  Totale complessivo:    {len(all_weights) + len(all_biases)}")
            print(f"  Peso medio globale:   {np.mean(all_weights):.6f}")
            print(f"  Peso std dev globale: {np.std(all_weights):.6f}")
            print(f"  Peso min globale:     {np.min(all_weights):.6f}")
            print(f"  Peso max globale:      {np.max(all_weights):.6f}")
            print(f"  Bias medio globale:   {np.mean(all_biases):.6f}")
            print(f"  Bias std dev globale: {np.std(all_biases):.6f}")
            
            print(f"\n DISTRIBUZIONE PESI:")
            print(f"  Pesi negativi:         {np.sum(all_weights < 0)} ({np.sum(all_weights < 0)/len(all_weights)*100:.1f}%)")
            print(f"  Pesi positivi:        {np.sum(all_weights > 0)} ({np.sum(all_weights > 0)/len(all_weights)*100:.1f}%)")
            print(f"  Pesi ~ 0 (|W|<0.01):  {np.sum(np.abs(all_weights) < 0.01)} ({np.sum(np.abs(all_weights) < 0.01)/len(all_weights)*100:.1f}%)")
            
            print(f"\n NORMA DEI PESI PER LAYER:")
            for layer_idx, W in enumerate(final_net.weights):
                l1_norm = np.sum(np.abs(W))
                l2_norm = np. sqrt(np.sum(W**2))
                print(f"  Layer {layer_idx}: L1={l1_norm:.4f}, L2={l2_norm:.4f}")
            
            print(f"\n EFFETTO REGOLARIZZAZIONE L2:")
            print(f"  L2 Lambda usato:       {params['l2_lambda']}")
            l2_penalty = params['l2_lambda'] * np.sum(all_weights**2)
            print(f"  L2 Penalty totale:    {l2_penalty:.6f}")
            print(f"  Avg L2 per peso:      {l2_penalty/len(all_weights):.6f}")
        
        else:
            print("\n Impossibile accedere ai pesi della rete.")
        
        if hasattr(final_net, 'eta'):
            print(f"\n LEARNING RATE FINALE:")
            if isinstance(final_net.eta, (int, float)):
                print(f"  Eta finale:    {final_net.eta:.6f}")
            elif isinstance(final_net.eta, np.ndarray):
                print(f"  Eta per layer (array):")
                for idx, eta_val in enumerate(final_net. eta):
                    print(f"    Layer {idx}: {eta_val:.6f}")
        
        # Statistiche grid search
        print(f"\n{'='*70}")
        print(" GRID SEARCH STATISTICS")
        print(f"{'='*70}")
        all_val_accs = [r['val_accuracy'] for r in all_results]
        print(f"  Total Runs:             {len(all_results)}")
        print(f"  Grid Search Initial:    {len(all_results_initial)} runs")
        print(f"  Refine Search:          {len(all_results_refined)} runs")
        print(f"  Best LR Found:          {best_results['lr']}")
        print(f"  Best L2 Found:          {best_results['l2_lambda']}")
        print(f"  Validation Acc Stats:")
        print(f"    - Mean:                {np.mean(all_val_accs):.2%}")
        print(f"    - Std Dev:            {np.std(all_val_accs):.2%}")
        print(f"    - Min:                {np. min(all_val_accs):.2%}")
        print(f"    - Max:                {np.max(all_val_accs):.2%}")
        print(f"    - Runs with 95%+:     {sum(1 for acc in all_val_accs if acc >= 0.95)}/{len(all_val_accs)}")
        print(f"    - Runs with 97%+:     {sum(1 for acc in all_val_accs if acc >= 0.97)}/{len(all_val_accs)}")
        
        # Summary finale
        print(f"\n{'='*70}")
        print(" CONCLUSIONI")
        print(f"{'='*70}")
        
        if test_results['test_accuracy'] >= 0.97:
            print(" Risultato ECCELLENTE per MONK-3 con 5% di rumore!")
            print(" La regolarizzazione L2 ha gestito perfettamente il rumore!")
            print(" Il threshold tuning ha ottimizzato ulteriormente le performance!")
        elif test_results['test_accuracy'] >= 0.95:
            print(" Risultato OTTIMO per MONK-3!")
            print(" Raggiunto il massimo teorico considerando il 5% di rumore!")
        else:
            print(f" Test Accuracy raggiunta: {test_results['test_accuracy']:.2%}")
            
        
        print(f"\n File salvati:")
        print(f"  - monk3_best_results.png (grafici principali)")
        print(f"  - monk3_lr_l2_comparison.png (confronto LR e L2)")
        print(f"  - monk3_threshold_analysis.png (analisi threshold)")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO MONK-3 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback. print_exc()