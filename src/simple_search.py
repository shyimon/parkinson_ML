# simple_search_fixed.py
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk3

def realistic_monk3_test():
    """Test realistico per Monk 3"""
    print("TEST REALISTICO PER MONK 3")
    print("="*60)
    
    # Carica i dati
    X_train, y_train, X_val, y_val, X_test, y_test = return_monk3(
        one_hot=True, 
        val_split=0.3,
        dataset_shuffle=True
    )
    
    print(f"Dimensioni dataset:")
    print(f"  Training:   {X_train.shape[0]} esempi")
    print(f"  Validation: {X_val.shape[0]} esempi")
    print(f"  Test:       {X_test.shape[0]} esempi")
    
    # Configurazione ottimizzata
    params = {
        'network_structure': [17, 8, 2],  # 17 input (one-hot), 8 hidden, 2 output
        'eta': 0.05,                      # Learning rate
        'l2_lambda': 0.001,               # Regolarizzazione
        'momentum': 0.9,                  # Momentum
        'algorithm': 'sgd', 'rprop'       # Algoritmo
        'activation_type': 'sigmoid',     # Attivazione
        'loss_type': 'binary_crossentropy', # Loss per classificazione binaria
        'weight_initializer': 'xavier',   # Inizializzazione
        'decay': 0.95,                    # Decay per LR scheduling
        'mu': 1.75,                       # Per quickprop
        'eta_plus': 1.2,                  # Per rprop
        'eta_minus': 0.5,                 # Per rprop
        'debug': False
    }
    
    print(f"\nParametri della rete: {params}")
    
    # Crea e addestra la rete
    net = NeuralNetwork(**params)
    
    print("\nAddestramento in corso...")
    history = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=500,
        batch_size=16,           # Mini-batch
        patience=25,            # Early stopping patience
        min_delta=0.0001,       # Minimo miglioramento richiesto
        verbose=True
    )
    
    # Valutazione
    print("\n" + "="*60)
    print("VALUTAZIONE FINALE")
    print("="*60)
    
    # Training set
    train_pred = net.predict(X_train)
    train_pred_class = (train_pred > 0.5).astype(int)
    train_acc = np.mean(train_pred_class == y_train)
    
    # Validation set
    val_pred = net.predict(X_val)
    val_pred_class = (val_pred > 0.5).astype(int)
    val_acc = np.mean(val_pred_class == y_val)
    
    # Test set
    test_pred = net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_acc = np.mean(test_pred_class == y_test)
    
    print(f"Training Accuracy:   {train_acc:.4%}")
    print(f"Validation Accuracy: {val_acc:.4%}")
    print(f"Test Accuracy:       {test_acc:.4%}")
    
    # Statistiche dettagliate
    print(f"\nStatistiche predizioni test:")
    print(f"  Min prediction: {test_pred.min():.6f}")
    print(f"  Max prediction: {test_pred.max():.6f}")
    print(f"  Mean prediction: {test_pred.mean():.6f}")
    print(f"  Std prediction: {test_pred.std():.6f}")
    
    # Confusion matrix
    tp = np.sum((test_pred_class == 1) & (y_test == 1))
    fp = np.sum((test_pred_class == 1) & (y_test == 0))
    tn = np.sum((test_pred_class == 0) & (y_test == 0))
    fn = np.sum((test_pred_class == 0) & (y_test == 1))
    
    print(f"\nConfusion Matrix (Test Set):")
    print(f"           Predicted 0  Predicted 1")
    print(f"Actual 0   {tn:10}  {fp:10}")
    print(f"Actual 1   {fn:10}  {tp:10}")
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetriche:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    # Grafico delle loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['training'], label='Training Loss', linewidth=2)
    plt.plot(history['validation'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Calcola accuracy approssimativa (1 - normalized loss)
    train_acc_history = 1 - np.array(history['training']) / np.max(history['training'])
    val_acc_history = 1 - np.array(history['validation']) / np.max(history['validation'])
    
    plt.plot(train_acc_history, label='Training (1 - norm loss)', alpha=0.7)
    plt.plot(val_acc_history, label='Validation (1 - norm loss)', alpha=0.7)
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Acc: {test_acc:.2%}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (approx)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monk3_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'test_accuracy': test_acc,
        'network': net,
        'history': history,
        'params': params
    }

if __name__ == "__main__":
    try:
        results = realistic_monk3_test()
        
        print(f"\n{'='*60}")
        print("RIEPILOGO FINALE")
        print("="*60)
        
        if results['test_accuracy'] >= 0.99:
            print("🎉🎉🎉 OBIETTIVO RAGGIUNTO: 99% accuracy! 🎉🎉🎉")
        elif results['test_accuracy'] >= 0.97:
            print("✅ Ottimo risultato! Vicino al massimo teorico (95-97% per Monk 3)")
        else:
            print(f"🔧 Accuracy: {results['test_accuracy']:.2%}")
            print("   Prova queste modifiche:")
            print("   1. Aumenta epoche a 500")
            print("   2. Prova learning rate 0.05")
            print("   3. Prova batch_size=16")
            print("   4. Aumenta hidden neurons a 12")
            print("   5. Prova algoritmo RPROP")
        
        print(f"\nParametri usati: {results['params']}")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()