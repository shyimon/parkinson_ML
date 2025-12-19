# simple_search_fixed.py
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import data_manipulation
from itertools import product
import pandas as pd

def _monk3_test():
    print("TEST REALISTICO PER MONK 3")
    print("="*60)
    
    # Carica i dati
    X_train, y_train, X_val, y_val, X_test, y_test = data_manipulation.return_monk3(
        one_hot=True, 
        val_split=0.5,
        dataset_shuffle=True
    )
    
    print(f"Dimensioni dataset:")
    print(f"  Training:   {X_train.shape[0]} esempi")
    print(f"  Validation: {X_val.shape[0]} esempi")
    print(f"  Test:       {X_test.shape[0]} esempi")

    '''grid = {
        'eta': [0.05, 0.1, 0.25, 0.5, 0.7],
        'l2_lambda': [0.0, 0.000001, 0.00001, 0.001, 0.01],
        'hidden_structure': [[3], [4], [6], [8],
                             [8, 4], [6, 2]],
        'momentum': [0.9],
        'algorithm': ['sgd'],
        'activation_type': ['sigmoid'],
        'loss_type': ['half_mse', 'mae', 'huber'],
        'weight_initializer': ['def', 'xavier'],
        'decay': [0.95, 0.9, 0.8],
        'batch_size': [1, 2, 4, 8, 16, 32],
        'patience': [10, 20, 50, 100]
    }'''

    grid = {
        'eta': [0.05, 0.5, 0.7],
        'l2_lambda': [0.0, 0.000001, 0.001],
        'hidden_structure': [[3], [4], [6], [8]],
        'momentum': [0.9],
        'algorithm': ['sgd'],
        'activation_type': ['sigmoid'],
        'loss_type': ['half_mse', 'mae', 'huber'],
        'weight_initializer': ['def', 'xavier'],
        'decay': [0.95, 0.9],
        'batch_size': [1, 2, 4, 16],
        'patience': [20, 50]
    }

    results = []
    params = product(grid['eta'], grid['l2_lambda'], grid['hidden_structure'], grid['momentum'], grid['algorithm'], grid['activation_type'], grid['loss_type'], grid['weight_initializer'], grid['decay'], grid['batch_size'], grid['patience'])
    total_runs = len(list(params))
    current_run = 1
    params = product(grid['eta'], grid['l2_lambda'], grid['hidden_structure'], grid['momentum'], grid['algorithm'], grid['activation_type'], grid['loss_type'], grid['weight_initializer'], grid['decay'], grid['batch_size'], grid['patience'])

    for eta, l2_lambda, hidden_structure, momentum, algorithm, activation_type, loss_type, weight_initializer, decay, batch_size, patience in params:
        print(f"\n\nTraining {current_run} of {total_runs}")
        current_run += 1

        network_structure = [X_test.shape[1]]
        network_structure.extend(hidden_structure)
        network_structure.append(y_test.shape[1])
        
        # Crea e addestra la rete
        net = NeuralNetwork(network_structure=network_structure,
                            eta=eta,
                            l2_lambda=l2_lambda,
                            momentum=momentum,
                            algorithm=algorithm,
                            activation_type=activation_type,
                            loss_type=loss_type,
                            weight_initializer=weight_initializer,
                            decay=decay)
        
        print("\nAddestramento in corso...")
        history = net.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=1000,
            batch_size=batch_size,           # Mini-batch
            patience=patience,            # Early stopping patience
            verbose=False
        )

        # Training set
        train_pred = net.predict(X_train)
        
        # Validation set
        val_pred = net.predict(X_val)

        train_loss = 0.5 * np.mean((train_pred - y_train)**2)
        val_loss = 0.5 * np.mean((val_pred - y_val)**2)

        results.append({
            'eta': eta,
            'l2_lambda': l2_lambda,
            'hidden_structure': hidden_structure,
            'momentum': momentum,
            'algorithm': algorithm,
            'activation_type': activation_type,
            'loss_type': loss_type,
            'weight_initializer': weight_initializer,
            'decay': decay,
            'batch_size': batch_size,
            'patience': patience,
            'val_loss': val_loss
        })
        
        print(f"Validation loss: {val_loss:.6f}")

        
        net.save_plots("img/plot.png")
        net.draw_network("img/network")
        
        results = sorted(results, key=lambda x: x['val_loss'])
    
    return results

if __name__ == "__main__":
    results = _monk3_test()
        
    print(f"\n{'='*60}")
    print("RIEPILOGO FINALE")
    print("="*60)
        
    print("\nTop configurations:")

    df = pd.DataFrame(results)
    df.to_csv("monk3_grid_search_results.csv", index=False)
    for r in results[:5]:
        print(r)











    '''

        # Valutazione
    print("\n" + "="*60)
    print("VALUTAZIONE FINALE")
    print("="*60)
    
    # Test set
    test_pred = net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_acc = np.mean(test_pred_class == y_test)
    
    print(f"Training Accuracy:   {train_acc:.4%}")
    print(f"Validation Accuracy: {val_acc:.4%}")
    print(f"Test Accuracy:       {test_acc:.4%}")
    
    # Confusion matrix
    tp = np.sum((test_pred_class == 1) & (y_test == 1))
    fp = np.sum((test_pred_class == 1) & (y_test == 0))
    tn = np.sum((test_pred_class == 0) & (y_test == 0))
    fn = np.sum((test_pred_class == 0) & (y_test == 1))
    
    print(f"\nConfusion Matrix (Test Set):")
    print(f"           Predicted 0  Predicted 1")
    print(f"Actual 0   {tn:10}  {fp:10}")
    print(f"Actual 1   {fn:10}  {tp:10}")
    '''