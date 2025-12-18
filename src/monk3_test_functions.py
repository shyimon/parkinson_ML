# monk3_test_functions.py
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_monk3

def run_monk3_test(params):
    """Esegue un test su Monk 3 con i parametri specificati"""
    print("TEST PER MONK 3")
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
    
    # Correzione: per classificazione binaria, output deve essere 1
    if params['network_structure'][-1] != 1:
        print(f"Correzione: Output layer modificato da {params['network_structure'][-1]} a 1")
        params['network_structure'][-1] = 1
    
    print(f"\nParametri della rete: {params}")
    
    # Crea e addestra la rete
    net = NeuralNetwork(**params)
    
    print("\nAddestramento in corso...")
    history = net.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=650,
        batch_size=16,
        patience=25,
        min_delta=0.0001,
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
    
    return {
        'test_accuracy': test_acc,
        'validation_accuracy': val_acc,
        'train_accuracy': train_acc,
        'network': net,
        'history': history,
        'params': params
    }