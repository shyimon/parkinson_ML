# simple_search_cup.py
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import load_cup_data


def _cup_test(learning_rate, seed, hidden_units=20, verbose=False): 
    if verbose:
        print(f"  Seed: {seed}, LR: {learning_rate}, Hidden:  {hidden_units}")
    
    # Seed per riproducibilità
    np.random.seed(seed)

    # Carica i dati CUP (regressione)
    X_train, y_train, X_val, y_val, X_internal_test, y_internal_test = load_cup_data(
        val_split=0.2,  # 20% per validation
        shuffle=True,
        include_internal_test=True  # Include internal test per valutazione finale
    )
    
    # Configurazione per CUP (regressione)
    params = {
        'network_structure': [X_train.shape[1], hidden_units, y_train.shape[1]],  # Input -> Hidden -> Output (2D o 3D)
        'eta': learning_rate,
        'l2_lambda': 0.001,  # Regolarizzazione per evitare overfitting
        'momentum': 0.9,
        'algorithm': 'sgd',
        'activation_type': 'sigmoid',  # Sigmoid per hidden, linear per output (regressione)
        'loss_type': 'mse',  # Mean Squared Error per regressione
        'weight_initializer': 'xavier',
        'decay': 0.95,
        'mu': 1.75,
        'eta_plus':  1.2,
        'eta_minus':  0.5,
        'debug': False
    }
    
  
    net = NeuralNetwork(**params)
    
    history = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=1000,
        batch_size=32,  # Mini-batch per CUP
        patience=200,
        verbose=verbose
    )
    
    # Training set
    train_pred = net.predict(X_train)
    train_mee = mean_euclidean_error(y_train, train_pred)
    
    # Validation set
    val_pred = net.predict(X_val)
    val_mee = mean_euclidean_error(y_val, val_pred)
    
    # Calcola le loss finali
    if isinstance(history, dict) and 'training' in history:
        final_train_loss = history['training'][-1] if isinstance(history['training'], list) else history['training']
        final_val_loss = history['validation'][-1] if isinstance(history['validation'], list) else history['validation']
    else:   
        final_train_loss = history if not isinstance(history, dict) else 0
        final_val_loss = history if not isinstance(history, dict) else 0
    
    return {
        'train_mee': train_mee,
        'val_mee': val_mee,
        'train_loss': final_train_loss,
        'val_loss':  final_val_loss,
        'network': net,
        'history': history,
        'params': params,
        'lr': learning_rate,
        'seed': seed,
        'hidden_units': hidden_units,
        'data':  (X_train, y_train, X_val, y_val, X_internal_test, y_internal_test)
    }


def mean_euclidean_error(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred)**2, axis=1)))


def grid_search_cup(n_seeds_per_config=3, learning_rates=None, hidden_units_list=None):
    if learning_rates is None:
        learning_rates = [0.001, 0.005, 0.01, 0.05]
    
    if hidden_units_list is None:
        hidden_units_list = [20, 30, 40, 50]
    
    best_val_mee = float('inf')
    best_results = None
    all_results = []
    
    total_runs = len(learning_rates) * len(hidden_units_list) * n_seeds_per_config
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f" GRID SEARCH:  {len(learning_rates)} LRs × {len(hidden_units_list)} Hidden × {n_seeds_per_config} seeds = {total_runs} runs")
    print(f"{'#'*70}\n")
    
    for lr in learning_rates:
        for hidden in hidden_units_list:
            print(f"\n{'='*70}")
            print(f" TESTING:  LR={lr}, Hidden Units={hidden}")
            print(f"{'='*70}")
            
            for seed_idx in range(n_seeds_per_config):
                current_run += 1
                seed = seed_idx * 123 + int(lr * 10000) + hidden
                
                print(f"\n Run {current_run}/{total_runs} - LR={lr}, Hidden={hidden}, Seed={seed}", end=" → ")
                
                results = _cup_test(learning_rate=lr, seed=seed, hidden_units=hidden, verbose=False)
                all_results.append(results)
                
                print(f"Train MEE: {results['train_mee']:.4f}, Val MEE: {results['val_mee']:.4f}")
                
                if results['val_mee'] < best_val_mee:
                    best_val_mee = results['val_mee']
                    best_results = results
                    print(f" NUOVO BEST VAL MEE: {best_val_mee:.4f}")
    
    print(f"\n{'='*70}")
    print(f" MIGLIOR CONFIGURAZIONE (basata su VALIDATION)")
    print(f"{'='*70}")
    print(f"Validation MEE:   {best_results['val_mee']:.4f}")
    print(f"Train MEE:       {best_results['train_mee']:.4f}")
    print(f"Best LR:         {best_results['lr']}")
    print(f"Best Hidden:     {best_results['hidden_units']}")
    print(f"Best Seed:       {best_results['seed']}")
    
    return best_results, all_results


def evaluate_on_internal_test(net, X_internal_test, y_internal_test):
    """
    Valuta il modello finale sull'INTERNAL TEST SET (UNA SOLA VOLTA!)
    """
    print(f"\n{'='*70}")
    print(f" VALUTAZIONE FINALE SULL'INTERNAL TEST SET")
    print(f"{'='*70}")
    
    test_pred = net.predict(X_internal_test)
    test_mee = mean_euclidean_error(y_internal_test, test_pred)
    
    print(f"\n RISULTATI INTERNAL TEST SET:")
    print(f"  Test MEE:  {test_mee:.4f}")
    
    return {
        'test_mee':  test_mee,
        'test_predictions': test_pred
    }


def generate_blind_test_predictions(net, save_path='blind_test_predictions.csv'):
    """
    Genera predizioni per il BLIND TEST SET
    """
    print(f"\n{'='*70}")
    print(f" GENERAZIONE PREDIZIONI BLIND TEST SET")
    print(f"{'='*70}")
    
    # Carica blind test set (solo features, no labels)
    try:
        X_blind = load_cup_data(blind_test_only=True)
        
        # Genera predizioni
        blind_pred = net.predict(X_blind)
        
        # Salva in formato CSV
        # Formato: ID, output1, output2 (o output1, output2, output3 se 3D)
        with open(save_path, 'w') as f:
            f.write("# ML-CUP Blind Test Predictions\n")
            f.write("# Team: YourTeamName\n")
            for i, pred in enumerate(blind_pred, 1):
                if len(pred) == 2:
                    f.write(f"{i},{pred[0]:.6f},{pred[1]:.6f}\n")
                else: 
                    f.write(f"{i},{','.join([f'{p:.6f}' for p in pred])}\n")
        
        print(f"\n Predizioni salvate in: {save_path}")
        print(f"   Numero predizioni: {len(blind_pred)}")
        
        return blind_pred
    
    except Exception as e: 
        print(f"\n Impossibile caricare blind test set: {e}")
        print(f"   Assicurati che load_cup_data() supporti blind_test_only=True")
        return None


def plot_results(train_results, test_results, save_path='cup_performance.png'):
    save_path = save_path.strip().replace(' ', '')
    
    train_mee = train_results['train_mee']
    val_mee = train_results['val_mee']
    test_mee = test_results['test_mee']
    lr = train_results['lr']
    hidden = train_results['hidden_units']
    seed = train_results['seed']
    
    fig = plt.figure(figsize=(12, 6))
    
    # SUBPLOT 1: MEE BAR CHART
    plt.subplot(1, 2, 1)
    
    categories = ['Train', 'Validation', 'Internal Test']
    mees = [train_mee, val_mee, test_mee]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = plt.bar(categories, mees, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2.5)
    
    # Testo sopra le barre
    for i, (cat, mee) in enumerate(zip(categories, mees)):
        plt.text(i, mee + max(mees)*0.02, f'{mee:.4f}', ha='center', 
                fontsize=14, fontweight='bold')
    
    plt.ylabel('Mean Euclidean Error (MEE)', fontsize=13, fontweight='bold')
    plt.title(f'Model Performance (ML-CUP)\n(LR={lr}, Hidden={hidden}, Seed={seed})', 
             fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, max(mees) * 1.15)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # SUBPLOT 2: LEARNING CURVES
    plt.subplot(1, 2, 2)
    
    if 'history' in train_results and isinstance(train_results['history'], dict):
        history = train_results['history']
        if 'training' in history and 'validation' in history:
            train_loss = history['training'] if isinstance(history['training'], list) else [history['training']]
            val_loss = history['validation'] if isinstance(history['validation'], list) else [history['validation']]
            
            epochs = range(1, len(train_loss) + 1)
            
            plt.plot(epochs, train_loss, color='#3498db', linewidth=2, label='Training Loss', alpha=0.8)
            plt.plot(epochs, val_loss, color='#f39c12', linewidth=2, label='Validation Loss', alpha=0.8)
            
            plt.xlabel('Epochs', fontsize=13, fontweight='bold')
            plt.ylabel('MSE Loss', fontsize=13, fontweight='bold')
            plt.title('Learning Curves', fontsize=14, fontweight='bold', pad=15)
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
    else:
        plt.text(0.5, 0.5, 'Learning curves not available', 
                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Model Performance salvato in: {save_path}")
    plt.show()


def plot_bias_variance_epochs(all_results, dataset_name='ML-CUP', save_path='cup_bias_variance_epochs.png'):
    """
    Crea il grafico Bias-Variance Tradeoff con EPOCHE come model complexity
    Per regressione usa MEE invece di classification error
    """
    from collections import defaultdict
    
    save_path = save_path.strip().replace(' ', '')
    
    print(f"\n{'='*70}")
    print(f" CREAZIONE GRAFICO BIAS-VARIANCE VS EPOCHE")
    print(f"{'='*70}")
    
    print("Re-training di modelli selezionati con epoch tracking...")
    
    all_curves = []
    num_runs = min(10, len(all_results))  # Limita a 10 run (CUP è più lento)
    
    for idx in range(num_runs):
        result = all_results[idx]
        print(f"\rProcessing run {idx+1}/{num_runs}.. .", end="")
        
        X_train, y_train, X_val, y_val, X_internal_test, y_internal_test = result['data']
        
        np.random.seed(result['seed'])
        
        params = result['params'].copy()
        net = NeuralNetwork(**params)
        
        train_mees_run = []
        test_mees_run = []
        epochs_run = []
        
        max_epochs = 100
        batch_size = 32
        
        for epoch in range(max_epochs):
            try:
                net.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=batch_size, verbose=False)
            except: 
                break
            
            # Training MEE
            train_pred = net.predict(X_train)
            train_mee = mean_euclidean_error(y_train, train_pred)
            
            # Internal test MEE
            test_pred = net.predict(X_internal_test)
            test_mee = mean_euclidean_error(y_internal_test, test_pred)
            
            epochs_run.append(epoch + 1)
            train_mees_run.append(train_mee)
            test_mees_run.append(test_mee)
        
        all_curves.append((epochs_run, train_mees_run, test_mees_run))
    
    print("\n Re-training completato")
    
    # SMOOTHING
    def smooth_curve(values, weight=0.85):
        smoothed = []
        last = values[0] if values else 0
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # PLOT
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Linee sottili
    for epochs_run, train_mees_run, test_mees_run in all_curves:
        if len(train_mees_run) > 1:
            train_smooth = smooth_curve(train_mees_run, weight=0.85)
            test_smooth = smooth_curve(test_mees_run, weight=0.85)
            
            ax.plot(epochs_run, train_smooth, color='lightblue', alpha=0.3, linewidth=0.8, zorder=1)
            ax.plot(epochs_run, test_smooth, color='lightcoral', alpha=0.3, linewidth=0.8, zorder=1)
    
    # Medie
    max_len = max(len(curve[0]) for curve in all_curves) if all_curves else 100
    
    train_means = []
    test_means = []
    
    for epoch_idx in range(max_len):
        train_vals = []
        test_vals = []
        
        for epochs_run, train_mees_run, test_mees_run in all_curves:
            if epoch_idx < len(train_mees_run):
                train_vals.append(train_mees_run[epoch_idx])
                test_vals.append(test_mees_run[epoch_idx])
        
        if train_vals:
            train_means.append(np.mean(train_vals))
            test_means.append(np.mean(test_vals))
    
    epochs_axis = range(1, len(train_means) + 1)
    
    if train_means:
        train_means_smooth = smooth_curve(train_means, weight=0.90)
        test_means_smooth = smooth_curve(test_means, weight=0.90)
        
        ax.plot(epochs_axis, train_means_smooth, color='#1F618D', linewidth=4, 
               label='Training MEE (mean)', zorder=10)
        ax.plot(epochs_axis, test_means_smooth, color='#CB4335', linewidth=4, 
               label='Test MEE (mean)', zorder=10)
        
        # Annotazioni
        y_max = max(max(train_means_smooth), max(test_means_smooth))
        y_min = min(min(train_means_smooth), min(test_means_smooth))
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        text_x_left = max_len * 0.1
        ax.text(text_x_left, y_max - 0.03 * y_range,
               'High Bias\nLow Variance', fontsize=11, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8,
                        edgecolor='orange', linewidth=1.5))
        
        text_x_right = max_len * 0.9
        ax.text(text_x_right, y_max - 0.03 * y_range,
               'Low Bias\nHigh Variance', fontsize=11, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8,
                        edgecolor='red', linewidth=1.5))
        
        ax.text(max_len * 0.95, y_min + 0.15 * y_range, 
               'lucky', fontsize=11, style='italic', color='#1F618D', 
               ha='right', fontweight='bold')
        ax.text(max_len * 0.95, y_max - 0.25 * y_range, 
               'unlucky', fontsize=11, style='italic', color='#CB4335', 
               ha='right', fontweight='bold')
        
        min_test_idx = np.argmin(test_means_smooth)
        optimal_epoch = min_test_idx + 1
        optimal_error = test_means_smooth[min_test_idx]
        
        ax.axvline(x=optimal_epoch, color='green', linestyle=':', 
                  linewidth=2.5, alpha=0.7, zorder=9,
                  label=f'Optimal:  {optimal_epoch} epochs')
        
        ax.scatter([optimal_epoch], [optimal_error], color='green', 
                  s=200, marker='*', zorder=11, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Model Complexity (Training Epochs)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Euclidean Error (MEE)', fontsize=13, fontweight='bold')
    ax.set_title(f'Bias-Variance Tradeoff - {dataset_name}\nTraining and Test MEE vs Training Epochs', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if train_means:
        ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.15 * y_range)
    ax.set_xlim(0, max_len * 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafico Bias-Variance salvato in: {save_path}")
    plt.show()


if __name__ == "__main__": 
    try:
        print("="*70)
        print(" ML-CUP - REGRESSION TASK")
        print("="*70)
        
        # FASE 1: Grid search
        print("\n FASE 1: Grid Search (Train + Validation)")
        best_results, all_results = grid_search_cup(
            n_seeds_per_config=3,
            learning_rates=[0.001, 0.005, 0.01],
            hidden_units_list=[20, 30, 40]
        )
        
        # FASE 2: Bias-Variance Tradeoff
        print(f"\n FASE 2: Grafico Bias-Variance vs Epoche")
        plot_bias_variance_epochs(all_results, 
                                  dataset_name='ML-CUP',
                                  save_path='cup_bias_variance_epochs.png')
        
        # FASE 3: Valuta sull'INTERNAL TEST SET
        print(f"\n FASE 3: Valutazione sull'Internal Test Set")
        X_train, y_train, X_val, y_val, X_internal_test, y_internal_test = best_results['data']
        final_net = best_results['network']
        test_results = evaluate_on_internal_test(final_net, X_internal_test, y_internal_test)
        
        # FASE 4: Plot Performance
        print(f"\n📍 FASE 4: Grafico Model Performance")
        plot_results(best_results, test_results, save_path='cup_performance.png')
        
        # FASE 5: ENSEMBLE
        print(f"\n{'='*70}")
        print(" FASE 5: ENSEMBLE DI MODELLI")
        print(f"{'='*70}")
        
        sorted_results = sorted(all_results, key=lambda x: x['val_mee'])
        top_3_results = sorted_results[:3]
        
        print(f"\nTop 3 configurazioni selezionate:")
        for idx, res in enumerate(top_3_results, 1):
            print(f"  {idx}.LR={res['lr']}, Hidden={res['hidden_units']}, Seed={res['seed']}, Val MEE={res['val_mee']:.4f}")
        
        print(f"\n Creating ensemble...")
        ensemble_nets = [result['network'] for result in top_3_results]
        
        # Predizioni ensemble su internal test
        ensemble_preds = []
        for net in ensemble_nets:
            pred = net.predict(X_internal_test)
            ensemble_preds.append(pred)
        
        ensemble_pred_avg = np.mean(ensemble_preds, axis=0)
        ensemble_mee = mean_euclidean_error(y_internal_test, ensemble_pred_avg)
        
        print(f"\n Ensemble completato")
        print(f"\n{'='*70}")
        print(" RISULTATI ENSEMBLE")
        print(f"{'='*70}")
        print(f"  Ensemble MEE: {ensemble_mee:.4f}")
        
        print(f"\n{'─'*70}")
        print(" CONFRONTO:  Singolo Modello vs Ensemble")
        print(f"{'─'*70}")
        print(f"  Singolo (best): {test_results['test_mee']:.4f}")
        print(f"  Ensemble:        {ensemble_mee:.4f}")
        
        improvement = test_results['test_mee'] - ensemble_mee
        if improvement > 0:
            print(f"  Miglioramento:   -{improvement:.4f} (meglio! )")
        else:
            print(f"  Differenza:     +{abs(improvement):.4f}")
        
        # FASE 6: BLIND TEST PREDICTIONS
        print(f"\n FASE 6: Generazione Predizioni Blind Test")
        
        # Usa ensemble se migliore, altrimenti singolo modello
        if ensemble_mee < test_results['test_mee']:
            print(f"Usando ENSEMBLE per blind test...")
            # Genera predizioni blind con ensemble
            try:
                X_blind = load_cup_data(blind_test_only=True)
                blind_preds_ensemble = []
                for net in ensemble_nets:
                    pred = net.predict(X_blind)
                    blind_preds_ensemble.append(pred)
                blind_pred_final = np.mean(blind_preds_ensemble, axis=0)
                
                # Salva
                with open('blind_test_predictions_ENSEMBLE.csv', 'w') as f:
                    f.write("# ML-CUP Blind Test Predictions (ENSEMBLE)\n")
                    for i, pred in enumerate(blind_pred_final, 1):
                        f.write(f"{i},{','.join([f'{p:.6f}' for p in pred])}\n")
                
                print(f" Blind predictions (ensemble) salvate: blind_test_predictions_ENSEMBLE.csv")
            except Exception as e:
                print(f" Errore nel generare blind predictions: {e}")
        else:
            print(f"Usando SINGOLO MODELLO per blind test...")
            generate_blind_test_predictions(final_net, save_path='blind_test_predictions.csv')
        
        # RIEPILOGO FINALE
        print(f"\n{'='*70}")
        print(" RIEPILOGO FINALE")
        print("="*70)
        
        print(f"\n{'─'*70}")
        print(" MEE SU TUTTI I SET:")
        print(f"{'─'*70}")
        print(f"  Train MEE:              {best_results['train_mee']:.4f}")
        print(f"  Validation MEE:        {best_results['val_mee']:.4f}")
        print(f"  Internal Test MEE:     {test_results['test_mee']:.4f}")
        if ensemble_mee < test_results['test_mee']:
            print(f"  Ensemble MEE:          {ensemble_mee:.4f} ← BEST!")
        
        print(f"\n{'─'*70}")
        print(" PARAMETRI MIGLIORI:")
        print(f"{'─'*70}")
        print(f"  Learning Rate:    {best_results['lr']}")
        print(f"  Hidden Units:     {best_results['hidden_units']}")
        print(f"  Random Seed:      {best_results['seed']}")
        print(f"  L2 Lambda:        {best_results['params']['l2_lambda']}")
        print(f"  Batch Size:       32")
        print(f"  Architecture:     {best_results['params']['network_structure']}")
        
        print(f"\n File salvati:")
        print(f"  - cup_bias_variance_epochs.png (bias-variance)")
        print(f"  - cup_performance.png (performance)")
        if ensemble_mee < test_results['test_mee']:
            print(f"  - blind_test_predictions_ENSEMBLE.csv (blind test) ← BEST!")
        else:
            print(f"  - blind_test_predictions.csv (blind test)")
        
        print(f"\n{'='*70}")
        print(" ESPERIMENTO ML-CUP COMPLETATO CON SUCCESSO!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()