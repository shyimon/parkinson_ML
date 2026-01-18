import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_CUP, normalize, MEE, MSE, denormalize
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def smooth_curve(values, weight=0.95): 
    """
    Smooth dei valori con exponential moving average
    Più weight è alto (vicino a 1), più la curva è liscia
    """
    if len(values) == 0:
        return []
    
    smoothed = []
    last = values[0]
    
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return smoothed


def train_best_config():
    """
    Training con la migliore configurazione trovata nella grid search
    """
    #  CONFIGURAZIONE 
    learning_rate = 0.005
    l2_lambda = 0.0005
    momentum = 0.9
    network_structure = [12, 60, 40, 4]
    seed = 345
    
    epochs = 1500
    batch_size = 32
    patience = 150
    
    print("\n" + "="*70)
    print(" TRAINING CUP - BEST CONFIGURATION")
    print("="*70)
    print(f"  Learning Rate:        {learning_rate}")
    print(f"  L2 Lambda:          {l2_lambda}")
    print(f"  Momentum:            {momentum}")
    print(f"  Architecture:       {network_structure}")
    print(f"  Batch Size:         {batch_size}")
    print(f"  Max Epochs:         {epochs}")
    print(f"  Patience:           {patience}")
    print(f"  Seed:               {seed}")
    print("="*70 + "\n")
    
    #  CARICA DATI 
    np.random.seed(seed)
    
    X_train, y_train, X_val, y_val, X_test, y_test = return_CUP(
        dataset_shuffle=True,
        train_size=350,
        validation_size=100,
        test_size=100
    )
    
    print(" Dataset:")
    print(f"   Training:      {X_train.shape[0]} samples")
    print(f"   Validation:   {X_val.shape[0]} samples")
    print(f"   Test:        {X_test.shape[0]} samples\n")
    
    #  NORMALIZZAZIONE 
    # Input:   [0, 1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    X_train_norm = normalize(X_train, 0, 1, x_min, x_max)
    X_val_norm = normalize(X_val, 0, 1, x_min, x_max)
    X_test_norm = normalize(X_test, 0, 1, x_min, x_max)
    
    # Target:  [-1, 1] per tanh
    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)
    y_train_norm = normalize(y_train, -1, 1, y_min, y_max)
    y_val_norm = normalize(y_val, -1, 1, y_min, y_max)
    y_test_norm = normalize(y_test, -1, 1, y_min, y_max)
    
    print("✓ Normalizzazione completata")
    print(f"   Input range:   [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    print(f"   Target range:  [{y_train_norm.min():.3f}, {y_train_norm.max():.3f}]\n")
    
    # CREA RETE NEURALE 
    params = {
        'network_structure': network_structure,
        'eta': learning_rate,
        'l2_lambda': l2_lambda,
        'momentum': momentum,
        'algorithm': 'sgd',
        'activation_type': 'tanh',
        'loss_type': 'mse',
        'weight_initializer': 'xavier',
        'decay': 0.98,
        'mu': 1.75,
        'eta_plus':  1.2,
        'eta_minus':  0.5,
        'debug': False
    }
    
    net = NeuralNetwork(**params)
    print(" Rete neurale creata\n")
    
    # INIZIALIZZAZIONE TRACKING 
    train_mse_history = []
    val_mse_history = []
    train_mee_history = []
    val_mee_history = []
    
    best_val_mee = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_weights_state = None
    
    def save_network_state(network):
        """Salva lo stato completo dei pesi della rete"""
        state = []
        for layer in network.layers:
            layer_state = []
            for neuron in layer:
                neuron_state = {
                    'weights': neuron.weights.copy() if hasattr(neuron, 'weights') else None,
                    'bias': neuron.bias if hasattr(neuron, 'bias') else None
                }
                layer_state.append(neuron_state)
            state.append(layer_state)
        return state
    
    def restore_network_state(network, state):
        """Ripristina lo stato completo dei pesi della rete"""
        for layer_idx, layer in enumerate(network.layers):
            for neuron_idx, neuron in enumerate(layer):
                neuron_data = state[layer_idx][neuron_idx]
                if neuron_data['weights'] is not None:
                    neuron.weights = neuron_data['weights'].copy()
                if neuron_data['bias'] is not None: 
                    neuron.bias = neuron_data['bias']
    
    # TRAINING LOOP
    print(" Inizio training.. .\n")
    
    for epoch in range(epochs):
        
        #  Shuffle dati 
        indices = np.random.permutation(len(X_train_norm))
        X_shuffled = X_train_norm[indices]
        y_shuffled = y_train_norm[indices]
        
        # Mini-batch training
        for batch_start in range(0, len(X_train_norm), batch_size):
            batch_end = min(batch_start + batch_size, len(X_train_norm))
            current_batch_size = batch_end - batch_start
            
            # Reset accumulatori gradienti
            net._reset_gradients()
            
            # Accumula gradienti nel batch
            for i in range(batch_start, batch_end):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                
                # Forward pass
                y_pred = net.forward(xi)
                
                # Backward pass (accumula gradienti)
                error_signal = net.compute_error_signal(yi, y_pred, loss_type='half_mse')
                error_signal = np.asarray(error_signal).flatten()
                net.backward(error_signal, accumulate=True)
            
            # Applica gradienti accumulati
            net._apply_accumulated_gradients(batch_size=current_batch_size)
        
        # Calcola MSE (su dati normalizzati) 
        train_pred_norm = net.predict(X_train_norm)
        train_mse = np.mean((train_pred_norm - y_train_norm) ** 2)
        train_mse_history.append(train_mse)
        
        val_pred_norm = net.predict(X_val_norm)
        val_mse = np.mean((val_pred_norm - y_val_norm) ** 2)
        val_mse_history.append(val_mse)
        
        # Calcola MEE (su dati denormalizzati) 
        train_pred = denormalize(train_pred_norm, -1, 1, y_min, y_max)
        train_mee = MEE(y_train, train_pred)
        train_mee_history.append(train_mee)
        
        val_pred = denormalize(val_pred_norm, -1, 1, y_min, y_max)
        val_mee = MEE(y_val, val_pred)
        val_mee_history.append(val_mee)
        
        #  Early stopping su validation MEE 
        if val_mee < best_val_mee:
            best_val_mee = val_mee
            best_epoch = epoch + 1
            patience_counter = 0
            best_weights_state = save_network_state(net)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping all'epoca {epoch + 1}")
            if best_weights_state is not None:
                restore_network_state(net, best_weights_state)
            break
        
        # Progress 
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch + 1}/{epochs}")
            print(f"  Train → MSE: {train_mse:.6f} | MEE: {train_mee:.4f}")
            print(f"  Val   → MSE: {val_mse:.6f} | MEE: {val_mee:.4f}")
    
    #  RISULTATI TRAINING 
    final_epoch = len(train_mse_history)
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETATO")
    print("="*70)
    print(f"  Epoche eseguite:      {final_epoch}")
    print(f"  Miglior epoca:      {best_epoch}")
    print(f"\n  Training (finale):")
    print(f"    MSE: {train_mse_history[-1]:.6f}")
    print(f"    MEE: {train_mee_history[-1]:.4f}")
    print(f"\n  Validation (finale):")
    print(f"    MSE: {val_mse_history[-1]:.6f}")
    print(f"    MEE: {val_mee_history[-1]:.4f}")
    print(f"\n  Best Validation MEE: {best_val_mee:.4f} (epoca {best_epoch})")
    
    #  VALUTAZIONE TEST SET 
    print("\n" + "="*70)
    print(" VALUTAZIONE TEST SET")
    print("="*70)
    
    test_pred_norm = net.predict(X_test_norm)
    test_pred = denormalize(test_pred_norm, -1, 1, y_min, y_max)
    
    test_mse = MSE(y_test, test_pred)
    test_mee = MEE(y_test, test_pred)
    
    print(f"  Test MSE: {test_mse:6f}")
    print(f"  Test MEE: {test_mee:.4f}")
    print("="*70 + "\n")
    
   
    return {
        'train_mse_history': train_mse_history,
        'val_mse_history': val_mse_history,
        'train_mee_history': train_mee_history,
        'val_mee_history': val_mee_history,
        'test_mse': test_mse,
        'test_mee': test_mee,
        'best_epoch': best_epoch,
        'network_structure': network_structure,
        'lr': learning_rate,
        'l2':  l2_lambda,
        'momentum': momentum
    }


def plot_loss_curves(results, save_mse='cup_loss_mse.png', save_mee='cup_loss_mee.png'):
    
   # Crea due grafici:  MSE e MEE vs Epochs
    
    train_mse = results['train_mse_history']
    val_mse = results['val_mse_history']
    train_mee = results['train_mee_history']
    val_mee = results['val_mee_history']
    best_epoch = results['best_epoch']
    
    epochs = range(1, len(train_mse) + 1)
    
    # Smoothing 
    train_mse_smooth = smooth_curve(train_mse, weight=0.95)
    val_mse_smooth = smooth_curve(val_mse, weight=0.95)
    train_mee_smooth = smooth_curve(train_mee, weight=0.95)
    val_mee_smooth = smooth_curve(val_mee, weight=0.95)
    
    # GRAFICO 1: MSE 
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # SOLO curve smoothed 
    ax.plot(epochs, train_mse_smooth, label='Training MSE',
            linewidth=3, color='#3498db')
    ax.plot(epochs, val_mse_smooth, label='Validation MSE',
            linewidth=3, color='#e74c3c')
    
    # Best epoch marker
    ax.axvline(x=best_epoch, color='green', linestyle='dotted',
               linewidth=2.5, alpha=0.7,
               label=f'Best epoch: {best_epoch}')
    
    if best_epoch <= len(val_mse_smooth):
        ax.scatter([best_epoch], [val_mse_smooth[best_epoch-1]],
                   color='gold', s=400, marker='*', zorder=11,
                   edgecolors='darkgreen', linewidths=3)
    
    ax.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax.set_title(f'Training & Validation MSE vs Epochs\n'
                 f'(LR={results["lr"]}, L2={results["l2"]}, '
                 f'Arch={results["network_structure"]})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_mse, dpi=150, bbox_inches='tight')
    print(f"✓ Grafico MSE salvato: {save_mse}")
    plt.show()
    
    # Grafico 2 MEE
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # SOLO curve smoothed 
    ax.plot(epochs, train_mee_smooth, label='Training MEE',
            linewidth=3, color='#9b59b6')
    ax.plot(epochs, val_mee_smooth, label='Validation MEE',
            linewidth=3, color='#e67e22')
    
    # Best epoch marker
    ax.axvline(x=best_epoch, color='green', linestyle='dotted',
               linewidth=2.5, alpha=0.7,
               label=f'Best epoch: {best_epoch}')
    
    if best_epoch <= len(val_mee_smooth):
        ax.scatter([best_epoch], [val_mee_smooth[best_epoch-1]],
                   color='gold', s=400, marker='*', zorder=11,
                   edgecolors='darkgreen', linewidths=3)
    
    ax.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('MEE', fontsize=13, fontweight='bold')
    ax.set_title(f'Training & Validation MEE vs Epochs\n'
                 f'(LR={results["lr"]}, L2={results["l2"]}, '
                 f'Arch={results["network_structure"]})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_mee, dpi=150, bbox_inches='tight')
    print(f"✓ Grafico MEE salvato: {save_mee}")
    plt.show()



if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print(" CUP - BEST CONFIGURATION TRAINING")
        print("="*70 + "\n")
        
        # Training
        results = train_best_config()
        
        # Plot grafici
        print("\n" + "="*70)
        print(" CREAZIONE GRAFICI")
        print("="*70 + "\n")
        plot_loss_curves(results)
        
        # Riepilogo finale
        print("\n" + "="*70)
        print("  RIEPILOGO FINALE")
        print("="*70)
        print("\n  LOSS FINALI:")
        print(f"    Training MSE:         {results['train_mse_history'][-1]:.6f}")
        print(f"    Validation MSE:     {results['val_mse_history'][-1]:.6f}")
        print(f"    Training MEE:       {results['train_mee_history'][-1]:.4f}")
        print(f"    Validation MEE:      {results['val_mee_history'][-1]:.4f}")
        print(f"    Test MSE:           {results['test_mse']:.6f}")
        print(f"    Test MEE:           {results['test_mee']:.4f}")
        
        print("\n  CONFIGURAZIONE:")
        print(f"    Learning Rate:       {results['lr']}")
        print(f"    L2 Lambda:          {results['l2']}")
        print(f"    Momentum:           {results['momentum']}")
        print(f"    Architecture:       {results['network_structure']}")
        print(f"    Best Epoch:         {results['best_epoch']}")
        
        print("\n" + "="*70)
        print("  COMPLETATO CON SUCCESSO!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()