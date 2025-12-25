# simple_search_fixed.py
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk1

def _monk1_test(learning_rate, seed, verbose=False): 
    print("TEST MONK 1")
    print("="*60)
    if verbose:
        print(f"   Seed: {seed}, LR: {learning_rate}")
    
    # Seed per riproducibilità
    np.random.seed(seed)

    # Carica i dati
    X_train, y_train, X_val, y_val, X_test, y_test = return_monk1(
        one_hot=True, 
        val_split=0.3,
        dataset_shuffle=True
    )
    
     # Configurazione con 4 neuroni nascosti
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
    
    print(f"\nParametri della rete: {params}")
    
    # Crea e addestra la rete
    net = NeuralNetwork(**params)
    
    print("\nAddestramento in corso...")
    history = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=350,
        batch_size=1,           # Mini-batch
        patience=100,            # Early stopping patience
       # min_delta=0.0001,       # Minimo miglioramento richiesto
        verbose=True
    )
    print("Addestramento completato.")

    # Valutazione
    print("\n" + "="*60)
    print("VALUTAZIONE FINALE")
    print("="*60)
    
    # Training set
    train_pred = net.predict(X_train)
    train_pred_class = (train_pred > 0.5).astype(int)
    train_acc = np.mean(train_pred_class == y_train)
    train_error = 1 - train_acc
    
    # Validation set
    val_pred = net.predict(X_val)
    val_pred_class = (val_pred > 0.5).astype(int)
    val_acc = np.mean(val_pred_class == y_val)
    val_error = 1 - val_acc
    
    # Test set
    test_pred = net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_acc = np.mean(test_pred_class == y_test)
    
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
        'network':  net,
        'history': history,
        'params': params,
        'lr': learning_rate,
        'seed': seed,
        'data':  (X_train, y_train, X_val, y_val, X_test, y_test)
    }
    
def grid_search_lr(n_seeds_per_lr=5, learning_rates=None):
    if learning_rates is None: 
        learning_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_val_acc = 0  # ← CAMBIATO:  usiamo validation accuracy! 
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
            
            results = _monk1_test(learning_rate=lr, seed=seed, verbose=False)
            all_results.append(results)
            
            print(f"Train:  {results['train_accuracy']:.2%}, Val: {results['val_accuracy']:.2%}")
            
            # SELEZIONE BASATA SU VALIDATION ACCURACY
            if results['val_accuracy'] > best_val_acc:
                best_val_acc = results['val_accuracy']
                best_results = results
                print(f"    NUOVO BEST VAL ACC: {best_val_acc:.2%}")
            
            # Se raggiungiamo 100% su validation, fermati
            if results['val_accuracy'] >= 1.0:
                print(f"\n 100% VALIDATION ACCURACY! Fermata anticipata.")
                break
        
        if best_val_acc >= 1.0:
            print(f"\n Obiettivo validation raggiunto!  Interrompo la ricerca.")
            break
    
    # Statistiche finali (SOLO train e validation)
    print(f"\n{'='*70}")
    print(f"🏆 MIGLIOR CONFIGURAZIONE (basata su VALIDATION)")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {best_results['val_accuracy']:.4%}")
    print(f"Train Accuracy:       {best_results['train_accuracy']:.4%}")
    print(f"Validation Error:    {best_results['val_error']:.4%}")
    print(f"Train Error:         {best_results['train_error']:.4%}")
    print(f"Best LR:             {best_results['lr']}")
    print(f"Best Seed:           {best_results['seed']}")
    
    # Statistiche aggregate
    all_val_accs = [r['val_accuracy'] for r in all_results]
    print(f"\n Statistiche VALIDATION su {len(all_results)} run:")
    print(f"  Media:     {np.mean(all_val_accs):.2%}")
    print(f"  Std Dev:  {np.std(all_val_accs):.2%}")
    print(f"  Min:      {np.min(all_val_accs):.2%}")
    print(f"  Max:      {np.max(all_val_accs):.2%}")
    print(f"  Runs con 100% val: {sum(1 for acc in all_val_accs if acc >= 1.0)}/{len(all_val_accs)}")
    
    return best_results, all_results


def retrain_and_track_errors(best_results):
    """
    Ri-addestra il modello migliore tracciando training error e VALIDATION error ad ogni epoca
    (NON test error!)
    
    Args:
        best_results: dizionario con i migliori risultati dalla grid search
    
    Returns: 
        train_errors: lista di training errors per epoca
        val_errors: lista di validation errors per epoca
    """
    print(f"\n{'='*70}")
    print(f" RI-ADDESTRAMENTO CON TRACKING DEGLI ERRORI")
    print(f"{'='*70}")
    print(f"LR: {best_results['lr']}, Seed: {best_results['seed']}")
    
    # Recupera i dati
    X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
    
    # Seed per riproducibilità
    np.random.seed(best_results['seed'])
    
    # Ricrea la rete con gli stessi parametri
    params = best_results['params']. copy()
    net = NeuralNetwork(**params)
    
    # Training manuale con tracking degli errori
    train_errors = []
    val_errors = []
    
    epochs = 350
    batch_size = 1
    patience = 150
    best_val_error = float('inf')
    patience_counter = 0
    
    print(f"\nAddestramento in corso (tracking errori ad ogni epoca)...")
    
    for epoch in range(epochs):
        # Training per un'epoca
        try:
            net.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=batch_size, verbose=False)
        except: 
            # Se fit() non supporta epoch=1, usa fit() completo una volta sola
            if epoch == 0:
                net.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, patience=patience, verbose=False)
            break
        
        # Calcola errori su train e VALIDATION (NON test!)
        train_pred = net.predict(X_train)
        train_pred_class = (train_pred > 0.5).astype(int)
        train_error = 1 - np.mean(train_pred_class == y_train)
        
        val_pred = net.predict(X_val)
        val_pred_class = (val_pred > 0.5).astype(int)
        val_error = 1 - np. mean(val_pred_class == y_val)
        
        train_errors.append(train_error)
        val_errors.append(val_error)
        
        # Early stopping basato su validation error
        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
        else: 
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Progress ogni 100 epoche
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch+1}/{epochs} - Train Error: {train_error:.4f}, Val Error: {val_error:.4f}")
    
    print(f"\n Training completato dopo {len(train_errors)} epoche")
    print(f"Final Train Error: {train_errors[-1]:.4%}")
    print(f"Final Val Error:    {val_errors[-1]:.4%}")
    
    return train_errors, val_errors, net


def evaluate_on_test_set(net, X_test, y_test):
    """
    Valuta il modello finale sul TEST SET (UNA SOLA VOLTA!)
    
    Args:
        net: rete neurale addestrata
        X_test:  test features
        y_test: test labels
    
    Returns: 
        dizionario con metriche sul test set
    """
    print(f"\n{'='*70}")
    print(f" VALUTAZIONE FINALE SUL TEST SET (prima volta! )")
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
    
    print(f"\n📊 RISULTATI TEST SET:")
    print(f"  Test Accuracy:  {test_acc:.4%}")
    print(f"  Test Error:     {test_error:.4%}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN: {fn}  TN: {tn}")
    
    return {
        'test_accuracy':  test_acc,
        'test_error': test_error,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }

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

def plot_results(train_results, train_errors, val_errors, test_results, save_path='monk1_results.png'):
    """
    Crea grafici completi dei risultati CON SMOOTHING
    
    Args:  
        train_results: risultati su train/val dalla grid search
        train_errors:   lista di training errors per epoca
        val_errors:  lista di validation errors per epoca
        test_results: risultati sul test set (calcolati UNA SOLA VOLTA)
        save_path: percorso dove salvare il grafico
    """
    train_acc = train_results['train_accuracy']
    val_acc = train_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    lr = train_results['lr']
    seed = train_results['seed']
    
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Training Error & Validation Error vs Epochs (CON SMOOTHING)
    plt.subplot(1, 3, 1)
    
    if train_errors is not None and val_errors is not None: 
        epochs = range(1, len(train_errors) + 1)
        
        # APPLICA SMOOTHING
        smoothing_weight = 0.85  # Più alto = più smooth (0.7-0.95)
        train_errors_smooth = smooth_curve(train_errors, weight=smoothing_weight)
        val_errors_smooth = smooth_curve(val_errors, weight=smoothing_weight)
        
        # Plot curve originali (trasparenti, per confronto)
        plt.plot(epochs, train_errors, linewidth=1, color='#3498db', alpha=0.2, label='_nolegend_')
        plt.plot(epochs, val_errors, linewidth=1, color='#e74c3c', alpha=0.2, label='_nolegend_')
        
        # Plot curve smoothate (principali)
        plt.plot(epochs, train_errors_smooth, label='Training Error (smoothed)', 
                linewidth=2.5, color='#3498db')
        plt.plot(epochs, val_errors_smooth, label='Validation Error (smoothed)', 
                linewidth=2.5, color='#e74c3c')
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Error Rate', fontsize=12, fontweight='bold')
        plt.title(f'Training & Validation Error vs Epochs\n(LR={lr}, Seed={seed})', 
                 fontsize=13, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Linea orizzontale allo 0% (obiettivo)
        plt.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.6)
        
        # Trova il minimo del validation error (sulla curva originale)
        min_val_error = np.min(val_errors)
        min_val_epoch = np.argmin(val_errors) + 1
        plt.scatter([min_val_epoch], [min_val_error], color='red', s=150, zorder=5, marker='*', 
                   edgecolors='black', linewidths=2, 
                   label=f'Min Val Error: {min_val_error:.2%} (epoch {min_val_epoch})')
        
        plt.legend(fontsize=9, loc='best')
        plt.ylim(-0.05, max(0.5, max(max(train_errors), max(val_errors)) * 1.1))
        
    else:
        plt.text(0.5, 0.5, 
                f"Training Error:     {train_results['train_error']:.2%}\n"
                f"Validation Error:  {train_results['val_error']:.2%}\n\n"
                f"(No epoch-by-epoch tracking)",
                ha='center', va='center', 
                transform=plt.gca().transAxes,
                fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        plt.title('Error Rates (Final Values)', fontsize=13, fontweight='bold')
    
    # Subplot 2: Accuracy Bar Chart (Train, Val, Test)
    plt.subplot(1, 3, 2)
    
    categories = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Aggiungi valori sopra le barre
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=13, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance\n(Test evaluated ONCE at the end)', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='100% Target')
    plt.legend(fontsize=10)
    
    # Subplot 3: Confusion Matrix (TEST SET)
    plt.subplot(1, 3, 3)
    
    cm = test_results['confusion_matrix']
    confusion_data = np.array([[cm['tn'], cm['fp']], 
                                [cm['fn'], cm['tp']]])
    
    im = plt.imshow(confusion_data, cmap='Blues', alpha=0.8)
    plt.colorbar(im, label='Count', fraction=0.046)
    
    # Aggiungi valori nelle celle
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
    plt.title(f'Confusion Matrix (TEST SET)\nPrecision:  {test_results["precision"]:.2%}, Recall: {test_results["recall"]:.2%}, F1: {test_results["f1"]:.2%}', 
              fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato in: {save_path}")
    plt.show()


def plot_lr_comparison(all_results, save_path='monk1_lr_comparison.png'):
    """
    Crea un grafico di confronto tra diversi learning rates
    BASATO SU VALIDATION ACCURACY (non test!)
    """
    # Raggruppa per learning rate
    lr_groups = {}
    for r in all_results:
        lr = r['lr']
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr]. append(r['val_accuracy'])  # ← VALIDATION! 
    
    # Ordina per learning rate
    lrs = sorted(lr_groups. keys())
    means = [np.mean(lr_groups[lr]) for lr in lrs]
    stds = [np.std(lr_groups[lr]) for lr in lrs]
    maxs = [np.max(lr_groups[lr]) for lr in lrs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Media con error bars
    ax1.errorbar(lrs, means, yerr=stds, marker='o', linewidth=2.5, markersize=10, 
                 capsize=5, capthick=2, label='Mean ± Std', color='#3498db')
    ax1.plot(lrs, maxs, marker='*', linewidth=2.5, markersize=14, 
             linestyle='--', color='#2ecc71', label='Max')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='100% Target')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Accuracy vs Learning Rate', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.5, 1.05)
    
    # Subplot 2: Box plot
    data_for_boxplot = [lr_groups[lr] for lr in lrs]
    bp = ax2.boxplot(data_for_boxplot, labels=[f'{lr}' for lr in lrs], patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='100% Target')
    ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim(0.5, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" Grafico di confronto salvato in: {save_path}")
    plt.show()


if __name__ == "__main__":  
    try:
        print("="*70)
        print(" MONK1 - RICERCA DEL 100% CON 4 HIDDEN NEURONS")
        print("="*70)
        
        # FASE 1: Grid search su learning rate (usa SOLO train e validation)
        print("\n FASE 1: Grid Search (Train + Validation)")
        best_results, all_results = grid_search_lr(
            n_seeds_per_lr=5,
            learning_rates=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        
        # FASE 2: Ri-addestra il modello migliore tracciando gli errori
        print(f"\n FASE 2: Re-training con Error Tracking (Train + Validation)")
        train_errors, val_errors, final_net = retrain_and_track_errors(best_results)
        
        # FASE 3: Valuta sul TEST SET (UNA SOLA VOLTA!)
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot dei risultati
        plot_results(best_results, train_errors, val_errors, test_results, save_path='monk1_best_results.png')
        plot_lr_comparison(all_results, save_path='monk1_lr_comparison.png')
        
        # ========================================================
        # RIEPILOGO FINALE CON PARAMETRI COMPLETI
        # ========================================================
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        # Status del risultato
        if test_results['test_accuracy'] >= 1.0:
            print(" PERFETTO:  100% TEST ACCURACY RAGGIUNTO!  🎉🎉🎉")
        elif test_results['test_accuracy'] >= 0.99:
            print(" ECCELLENTE: 99%+ test accuracy!")
        elif test_results['test_accuracy'] >= 0.95:
            print("✓ OTTIMO: 95%+ test accuracy!")
        else:
            print(f" Test Accuracy: {test_results['test_accuracy']:.2%}")
        
        # Accuracy su tutti i set
        print(f"\n{'─'*70}")
        print(" ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Accuracy:        {best_results['train_accuracy']:.4%}")
        print(f"  Validation Accuracy:  {best_results['val_accuracy']:.4%}")
        print(f"  Test Accuracy (FINAL): {test_results['test_accuracy']:.4%}")
        
        # Metriche dettagliate
        print(f"\n{'─'*70}")
        print(" METRICHE DETTAGLIATE (TEST SET):")
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
        
        # PARAMETRI COMPLETI
        params = best_results['params']
        print(f"\n{'='*70}")
        print("  PARAMETRI DELLA RETE NEURALE")
        print(f"{'='*70}")
        
        print(f"\n  ARCHITETTURA:")
        print(f"  Network Structure:      {params['network_structure']}")
        print(f"    - Input Layer:       {params['network_structure'][0]} neurons")
        print(f"    - Hidden Layer(s):   {params['network_structure'][1:-1]} neurons")
        print(f"    - Output Layer:      {params['network_structure'][-1]} neuron(s)")
        
        print(f"\n TRAINING:")
        print(f"  Algorithm:             {params['algorithm']. upper()}")
        print(f"  Learning Rate (eta):   {params['eta']}")
        print(f"  Momentum:              {params['momentum']}")
        print(f"  Batch Size:            1 (SGD online)")
        print(f"  Max Epochs:            1500")
        print(f"  Early Stopping Patience: 150")
        print(f"  Actual Epochs Trained: {len(train_errors)}")
        
        print(f"\n REGOLARIZZAZIONE:")
        print(f"  L2 Lambda:             {params['l2_lambda']}")
        print(f"  Decay:                  {params['decay']}")
        
        print(f"\n FUNZIONI:")
        print(f"  Activation Function:   {params['activation_type']}")
        print(f"  Loss Function:         {params['loss_type']}")
        print(f"  Weight Initializer:    {params['weight_initializer']}")
        
        print(f"\n RIPRODUCIBILITÀ:")
        print(f"  Random Seed:           {best_results['seed']}")
        
        print(f"\n DATASET:")
        print(f"  Training Set:          {X_train.shape[0]} esempi")
        print(f"  Validation Set:        {X_val.shape[0]} esempi")
        print(f"  Test Set:              {X_test.shape[0]} esempi")
        print(f"  Validation Split:      30%")
        print(f"  Dataset Shuffle:       True")
        print(f"  One-Hot Encoding:      True")
        
        # Parametri aggiuntivi (per algoritmi specifici)
        if params['algorithm'] == 'quickprop':
            print(f"\n PARAMETRI QUICKPROP:")
            print(f"  Mu:                    {params. get('mu', 'N/A')}")
        elif params['algorithm'] == 'rprop':
            print(f"\n PARAMETRI RPROP:")
            print(f"  Eta Plus:              {params.get('eta_plus', 'N/A')}")
            print(f"  Eta Minus:             {params.get('eta_minus', 'N/A')}")
        
        # Statistiche della grid search
        print(f"\n{'='*70}")
        print(" GRID SEARCH STATISTICS")
        print(f"{'='*70}")
        all_val_accs = [r['val_accuracy'] for r in all_results]
        print(f"  Total Runs:            {len(all_results)}")
        print(f"  Learning Rates Tested:  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]")
        print(f"  Seeds per LR:          5")
        print(f"  Best LR Found:         {best_results['lr']}")
        print(f"  Validation Acc Stats:")
        print(f"    - Mean:               {np.mean(all_val_accs):.2%}")
        print(f"    - Std Dev:           {np.std(all_val_accs):.2%}")
        print(f"    - Min:               {np. min(all_val_accs):.2%}")
        print(f"    - Max:               {np.max(all_val_accs):.2%}")
        print(f"    - Runs with 100%:    {sum(1 for acc in all_val_accs if acc >= 1.0)}/{len(all_val_accs)}")
        
        # Summary finale
        print(f"\n{'='*70}")
        print(" CONCLUSIONI")
        print(f"{'='*70}")
        
        if test_results['test_accuracy'] >= 1.0:
            print("La rete ha raggiunto il 100% su Train, Validation E Test!")
            print(" Classificazione perfetta senza errori!")
            print("Configurazione ottimale trovata con successo!")
        else:
            print(f" Test Accuracy raggiunta: {test_results['test_accuracy']:.2%}")
            
        
        print(f"\n File salvati:")
        print(f"  - monk1_best_results.png (grafici principali)")
        print(f"  - monk1_lr_comparison.png (confronto learning rates)")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:  
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()