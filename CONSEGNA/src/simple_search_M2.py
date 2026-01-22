import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk2


def _monk2_test(learning_rate, seed, verbose=False): 
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}")
    
    # Seed per riproducibilità
    np.random.seed(seed)

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
    
    best_val_loss = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=2000,  
        batch_size=1,
        patience=200,
        verbose=verbose
    )
    
    # Ottieni la loss_history e accuracy_history completa dalla rete
    loss_history = net.loss_history
    accuracy_history = net.accuracy_history
    
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
    final_train_loss = loss_history['training'][-1]
    final_val_loss = loss_history['validation'][-1]
    
    return {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_error': train_error,
        'val_error': val_error,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'network': net,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'params': params,
        'lr': learning_rate,
        'seed': seed,
        'data': (X_train, y_train, X_val, y_val, X_test, y_test)
    }


def grid_search_lr(n_seeds_per_lr=10, learning_rates=None):
    if learning_rates is None: 
        learning_rates = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
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
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print(f"\n Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN: {fn}  TN: {tn}")
    
    return {
        'test_accuracy': test_acc,
        'test_error':  test_error,
        'test_half_mse': test_half_mse,
        'precision': precision,
        'recall':  recall,
        'f1':  f1,
        'confusion_matrix': {'tp': tp, 'fp':  fp, 'tn': tn, 'fn': fn}
    }


def plot_results(train_results, test_results, save_path='monk2_performance.png'):
    # Crea Confusion Matrix e stampa accuracy

    save_path = save_path.strip().replace(' ', '')
    
    train_acc = train_results['train_accuracy']
    val_acc = train_results['val_accuracy']
    test_acc = test_results['test_accuracy']
    lr = train_results['lr']
    seed = train_results['seed']
    
    # stampa accuracy
    print(f"\n{'─'*70}")
    print(" ACCURACY FINALI:")
    print(f"{'─'*70}")
    print(f"  Train:        {train_acc:.4%}")
    print(f"  Validation:  {val_acc:.4%}")
    print(f"  Test:        {test_acc:.4%}")

    # CONFUSION MATRIX
    fig = plt.figure(figsize=(8, 7))
    
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

def plot_loss_vs_epochs(result, save_path='monk2_loss_vs_epochs.png'):
    """
    Grafico della Half MSE Loss in funzione delle epoche per Train e Validation
    """
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f"  CREAZIONE GRAFICO Loss vs epochs (Half MSE)")
    print(f"{'='*70}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = result['data']
    np.random.seed(result['seed'])
    
    params = result['params'].copy()
    net = NeuralNetwork(**params)
    
    train_losses = []
    val_losses = []
    epochs_list = []
    
    max_epochs = 2000
    batch_size = 1
    patience = 200
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Re-training del modello migliore con tracking delle loss...")
    
     # Reset learning per training da zero
    net.best_val_loss = np.inf
    net.early_stop_wait = 0
    net.lr_wait = 0

    for epoch in range(max_epochs):
        # Salva lo stato prima del training
        old_patience = 200
    
        # Training per una singola epoca (senza early stopping)
        net.early_stop_wait = 0  # Reset per evitare early stop
    
        epoch_train_loss = 0.0
        for i in range(len(X_train)):
            xi = X_train[i]
            yi = y_train[i]
        
            y_pred = net.forward(xi)
            sample_loss = net.compute_loss(yi, y_pred, loss_type=net.loss_type)
            epoch_train_loss += np.sum(sample_loss)
        
            err = net.compute_error_signal(yi, y_pred, loss_type=net.loss_type)
            err = np.asarray(err).flatten()
            net.backward(err, accumulate=False)  
    
        # Calcola loss medie (predizione su tutto il set)
        train_pred = net.predict(X_train)
        train_loss = np.sum(net.compute_loss(y_train, train_pred, loss_type=net.loss_type)) / len(y_train)
    
        val_pred = net.predict(X_val)
        val_loss = np.sum(net.compute_loss(y_val, val_pred, loss_type=net.loss_type)) / len(y_val)
    
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else: 
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoca {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print(f" Re-training completato:  {len(epochs_list)} epoche")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs_list, train_losses, color='#1F618D', linewidth=2.5, 
           label='Training Loss (Half MSE)', alpha=0.9)
    ax.plot(epochs_list, val_losses, color='#CB4335', linewidth=2.5, 
           label='Validation Loss (Half MSE)', alpha=0.9)
    
    min_val_idx = np.argmin(val_losses)
    min_val_epoch = epochs_list[min_val_idx]
    min_val_loss = val_losses[min_val_idx]
    
    ax.axvline(x=min_val_epoch, color='green', linestyle=':',
              linewidth=2.5, alpha=0.7, label=f'Min Val Loss @ epoch {min_val_epoch}')
    ax.scatter([min_val_epoch], [min_val_loss], color='green', 
              s=200, marker='*', zorder=11, edgecolors='black', linewidths=2,
              label=f'Min Val Loss:  {min_val_loss:.6f}')
    
    ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Half MSE Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Validation Loss vs Epochs (MONK-2)\nLR={result["lr"]}, Seed={result["seed"]}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, max(epochs_list) * 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grafico Loss vs epochs salvato in: {save_path}")
    plt.show()
    
    return net, train_losses, val_losses


def plot_accuracy_vs_epochs(result, save_path='monk2_accuracy_vs_epochs.png'):
    """
    NUOVO: Grafico dell'Accuracy in funzione delle epoche per Train e Validation
    """
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f"  CREAZIONE GRAFICO Accuracy vs Epoche")
    print(f"{'='*70}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = result['data']
    np.random.seed(result['seed'])
    
    params = result['params'].copy()
    net = NeuralNetwork(**params)
    
    train_accuracies = []
    val_accuracies = []
    epochs_list = []
    
    max_epochs = 2000
    batch_size = 1
    patience = 200
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Re-training del modello migliore con tracking delle accuracy...")
    
    for epoch in range(max_epochs):
        try:
            net.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=batch_size, verbose=False)
        except: 
            break
        
        train_pred = net.predict(X_train)
        train_pred_class = (train_pred > 0.5).astype(int)
        train_acc = np.mean(train_pred_class == y_train)
        
        val_pred = net.predict(X_val)
        val_pred_class = (val_pred > 0.5).astype(int)
        val_acc = np.mean(val_pred_class == y_val)
        
        val_loss = 0.5 * np.mean((val_pred - y_val)**2)
        
        epochs_list.append(epoch + 1)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoca {epoch + 1}/{max_epochs} - Train Acc: {train_acc:.4%}, Val Acc: {val_acc:.4%}")
    
    print(f" Re-training completato: {len(epochs_list)} epoche")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs_list, train_accuracies, color='#3498db', linewidth=2.5, 
           label='Training Accuracy', alpha=0.9)
    ax.plot(epochs_list, val_accuracies, color='#e67e22', linewidth=2.5, 
           label='Validation Accuracy', alpha=0.9)
    
    max_val_idx = np.argmax(val_accuracies)
    max_val_epoch = epochs_list[max_val_idx]
    max_val_acc = val_accuracies[max_val_idx]
    
    ax.axvline(x=max_val_epoch, color='green', linestyle=':', 
              linewidth=2.5, alpha=0.7, label=f'Max Val Acc @ epoch {max_val_epoch}')
    ax.scatter([max_val_epoch], [max_val_acc], color='green', 
              s=200, marker='*', zorder=11, edgecolors='black', linewidths=2,
              label=f'Max Val Acc: {max_val_acc:.2%}')
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
               alpha=0.5, label='100% Target')
    
    ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Validation Accuracy vs Epochs (MONK-2)\nLR={result["lr"]}, Seed={result["seed"]}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, max(epochs_list) * 1.02)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" Grafico Accuracy vs Epochs salvato in: {save_path}")
    plt.show()
    
    return net, train_accuracies, val_accuracies

def plot_loss_epochs(all_results, dataset_name='MONK-2', save_path='monk2_loss_epochs.png'):
    from collections import defaultdict
    
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO Loss vs Epoche")
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
        params = result['params'].copy()
        net = NeuralNetwork(**params)
        
        train_losses_run = []
        val_losses_run = []
        epochs_run = []
        
        max_epochs = 200
        batch_size = 1
        
        for epoch in range(max_epochs):
            epoch_train_loss = 0.0
            for start_idx in range(0, len(X_train), batch_size):
                end_idx=min(start_idx+batch_size, len(X_train))
                current_batch_size = end_idx - start_idx

                net._reset_gradients

                for i in range(start_idx, end_idx):
                    xi= X_train[i]
                    yi = y_train[i]
                    y_pred = net.forward(xi)

                    sample_loss = net.compute_loss(yi, y_pred, loss_type=net.loss_type)
                    epoch_train_loss += np.sum(sample_loss)
            
                    err = net.compute_error_signal(yi, y_pred, loss_type=net.loss_type)
                    err = np.asarray(err).flatten()
                    net.backward(err, accumulate=False)
            
            # Training error
            train_loss = epoch_train_loss / len(X_train)

            # Validation error 
            val_pred = net.predict(X_val)
            val_loss = np.sum(net.compute_loss(y_val, val_pred, loss_type=net.loss_type)) / len(y_val)
            
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
        train_means_smooth = smooth_curve(train_means, weight=0.95)
        test_means_smooth = smooth_curve(test_means, weight=0.95)
        
        # LINEE  (medie smoothed)
        ax.plot(epochs_axis, train_means_smooth, color='#1F618D', linewidth=2, 
               label='Training Loss (mean)', zorder=10)
        
        ax.plot(epochs_axis, test_means_smooth, color='#CB4335', linewidth=2, 
               label='Validation Loss (mean)', zorder=10)
        
        # ANNOTAZIONI
        y_max = max(max(train_means_smooth), max(test_means_smooth))
        y_min = min(min(train_means_smooth), min(test_means_smooth))
        y_range = y_max - y_min if y_max > y_min else 1.0
        
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
    ax.set_xlabel('Epochs)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Half MSE', fontsize=13, fontweight='bold')
    ax.set_title(f'Loss vs Epochs - {dataset_name}\n', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if train_means:
        ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.15 * y_range)
    ax.set_xlim(0, max_len * 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Loss vs epochs salvato in: {save_path}")
    plt.show()

def plot_accuracy_vs_epochs(all_results, dataset_name='MONK-2', save_path='monk2_accuracy_vs_epochs.png'):
    """
    Grafico dell'Accuracy in funzione delle epoche per Train e Validation
    """
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f"  CREAZIONE GRAFICO Accuracy vs Epoche")
    print(f"{'='*70}")
    
    # Filtra per modelli con performance eccellente
    perfect_results = [r for r in all_results if r['val_accuracy'] >= 0.999]
    high_perf_results = [r for r in all_results if r['val_accuracy'] >= 0.98]
    
    if len(perfect_results) >= 5:
        print(f" Usando {len(perfect_results)} modelli con val_acc >= 99.9%")
        results_to_use = perfect_results
    elif len(high_perf_results) >= 5:
        print(f" Usando {len(high_perf_results)} modelli con val_acc >= 98%")
        results_to_use = high_perf_results
    else:
        print(f"  Usando tutti i {len(all_results)} modelli disponibili")
        results_to_use = all_results
    
    # Ordina per validation accuracy
    results_to_use = sorted(results_to_use, key=lambda x: x['val_accuracy'], reverse=True)
    
    # USA LE HISTORY GIÀ SALVATE - Filtra solo modelli che raggiungono 100%
    print(f"Elaborazione history dai modelli che raggiungono 100%...")
    
    perfect_models = []
    for result in results_to_use[:30]:
        accuracy_history = result['accuracy_history']
        val_accs = accuracy_history['validation']
        if max(val_accs) >= 0.9999:
            perfect_models.append(result)
    
    if len(perfect_models) == 0:
        print(f"  Nessun modello raggiunge 100%. Uso i migliori disponibili")
        perfect_models = results_to_use[:10]
    else:
        print(f"  Trovati {len(perfect_models)} modelli che raggiungono 100%")
    
    models_to_use = perfect_models[:10]
    print(f"  Usando {len(models_to_use)} modelli per il grafico")
    
    all_curves = []
    for idx, result in enumerate(models_to_use):
        accuracy_history = result['accuracy_history']
        
        train_accs_run = accuracy_history['training']
        val_accs_run = accuracy_history['validation']
        epochs_run = list(range(1, len(train_accs_run) + 1))
        
        if idx == 0:
            max_val_acc = max(val_accs_run) if val_accs_run else 0
            print(f"  Esempio modello 1: Max val_acc = {max_val_acc:.4%}")
        
        all_curves.append((epochs_run, train_accs_run, val_accs_run))
    
    print("\nElaborazione completata")
    
    # SMOOTHING
    def smooth_curve(values, weight=0.92):
        smoothed = []
        last = values[0] if values else 0
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # CALCOLA MEDIE
    max_len = max(len(curve[0]) for curve in all_curves) if all_curves else 200
    
    train_acc_means = []
    val_acc_means = []
    
    for epoch_idx in range(max_len):
        train_vals = []
        val_vals = []
        
        for epochs_run, train_accs, val_accs in all_curves: 
            if epoch_idx < len(train_accs):
                train_vals.append(train_accs[epoch_idx])
                val_vals.append(val_accs[epoch_idx])
        
        if train_vals:
            train_acc_means.append(np.mean(train_vals))
            val_acc_means.append(np.mean(val_vals))
    
    epochs_axis = range(1, len(train_acc_means) + 1)
    
    # PLOT
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if train_acc_means:
        # Applica smoothing
        train_acc_smooth = smooth_curve(train_acc_means, weight=0.92)
        val_acc_smooth = smooth_curve(val_acc_means, weight=0.92)
        
        ax.plot(epochs_axis, train_acc_smooth, color='#1F618D', linewidth=2.5,
               label='Training Accuracy', alpha=0.9)
        ax.plot(epochs_axis, val_acc_smooth, color='#CB4335', linewidth=2.5,
               label='Validation Accuracy', alpha=0.9)
        
        # Optimal point
        max_val_idx = np.argmax(val_acc_smooth)
        optimal_epoch = list(epochs_axis)[max_val_idx]
        optimal_acc = val_acc_smooth[max_val_idx]
        
        ax.axvline(x=optimal_epoch, color='green', linestyle=':',
                  linewidth=2.5, alpha=0.7,
                  label=f'Optimal:  {optimal_epoch} epochs')
        ax.scatter([optimal_epoch], [optimal_acc], color='green',
                  s=200, marker='*', zorder=11, edgecolors='black', linewidths=2)
        
        # Target 100%
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                   alpha=0.5, label='100% Target')
        
        # Stampa statistiche
        max_raw_val_acc = max(val_acc_means)
        print(f"\n  Max Validation Accuracy (raw): {max_raw_val_acc:.4%}")
        print(f"  Max Validation Accuracy (smoothed): {optimal_acc:.4%}")
    
    ax.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(f'Training and Validation Accuracy vs Epochs - {dataset_name}',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, max_len * 1.02)
    ax.set_ylim(0, 1.05)
    
    # Formatta asse y come percentuale
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" Grafico Accuracy vs Epochs salvato in: {save_path}")
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
            learning_rates=[0.001, 0.002, 0.005, 0.003]
        )

        print(f"\n Usando tutte le {len(all_results)} configurazioni per il grafico\n")
       # Salva risultati originali
        all_results_original = all_results.copy()

        # Filtra modelli NON perfetti SOLO loss-epochs
        imperfect_results = [r for r in all_results if r['val_accuracy'] < 1.0 or r['train_accuracy'] < 1.0]

        if len(imperfect_results) >= 5:
            results_for_loss_epochs = imperfect_results
        else:
            results_for_loss_epochs = all_results

        # FASE 2: Loss vs Epochs (modelli sub-ottimali)
        plot_loss_epochs(results_for_loss_epochs,
                          dataset_name='MONK-2',
                          save_path='monk2_loss_epochs.png')

        # FASE 2b: Accuracy vs Epoche (modelli ottimali - ORIGINALI)
        plot_accuracy_vs_epochs(all_results_original,  #  Usa modelli ottimali
                        dataset_name='MONK-2',
                        save_path='monk2_accuracy_vs_epochs.png')
        
        # FASE 3: Valuta sul TEST SET (UNA SOLA VOLTA!)
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Model Performance + Confusion Matrix
        print(f"\n FASE 4: Grafico Model Performance")
        plot_results(best_results, test_results, save_path='monk2_performance.png')
        
        
        # FASE 5: Valuta sul TEST SET (UNA SOLA VOLTA!)
        print(f"\n FASE 3: Valutazione finale sul Test Set")
        X_train, y_train, X_val, y_val, X_test, y_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_test_set(final_net, X_test, y_test)
        
        # FASE 4: Plot Model Performance + Confusion Matrix
        print(f"\n FASE 4: Grafico Model Performance")
        plot_results(best_results, test_results, save_path='monk2_performance.png')
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        if test_results['test_accuracy'] >= 1.0:
            print("  PERFETTO:   100% TEST ACCURACY RAGGIUNTO!")
        elif test_results['test_accuracy'] >= 0.95:
            print("  ECCELLENTE: 95%+ test accuracy!")
        elif test_results['test_accuracy'] >= 0.90:
            print("  OTTIMO:  90%+ test accuracy!")
        else:
            print(f" Test Accuracy: {test_results['test_accuracy']:.2%}")
        
        print(f"\n{'─'*70}")
        print(" ACCURACY SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train Accuracy:         {best_results['train_accuracy']:.4%}")
        print(f"  Validation Accuracy:     {best_results['val_accuracy']:.4%}")
        print(f"  Test Accuracy (FINAL):  {test_results['test_accuracy']:.4%}")
        
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
        print(f"  L2 Lambda:       {best_results['params']['l2_lambda']}")
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
        print(f"  - monk2_loss_epochs.png (loss vs epochs)")
        print(f"  - monk2_accuracy_vs_epochs.png (accuracy vs epochs)")
        print(f"  - monk2_performance.png (confusion matrix)")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO MONK-2 COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:  
        print(f"\n ERRORE:  {e}")
        import traceback
        traceback.print_exc()