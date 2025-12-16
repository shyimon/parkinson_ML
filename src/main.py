import numpy as np
import data_manipulation as data
import grid_search

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk3(one_hot=True, 
                            dataset_shuffle=True)
    
    print(f"Shape dei dati:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Stampa il primo esempio per verifica
    print(f"\nPrimo esempio X_train[0]:")
    print(f"  Valori: {X_train[0]}")
    print(f"  Tipo: {type(X_train[0])}")
    print(f"  Dtype: {X_train[0].dtype}")
    
    # Verifica se ci sono NaN
    print(f"\nControllo NaN:")
    print(f"  NaN in X_train: {np.isnan(X_train).any()}")
    print(f"  NaN in y_train: {np.isnan(y_train).any()}")

    # Normalizzo
    X_train_norm, X_val_norm, X_test_norm = data.normalize_dataset(
            X_train, X_val, X_test, 0, 1
    )
   
     # Esegui ricerca dicotomica
    results = grid_search.run_advanced_monk_search(3)
    
    # uso i parametri migliori per un training finale
    best_params = results['best_params']
    
    print("\n" + "="*60)
    print("CONFIGURAZIONE CONSIGLIATA:")
    print("="*60)
    
    # Costruisco la struttura della rete
    input_size = X_train.shape[1]
    output_size = 1
    
    network_structure = [input_size]
    if 'hidden_structure' in best_params:
        network_structure.extend(best_params['hidden_structure'])
    network_structure.append(output_size)
    
    print(f"\nStruttura rete: {network_structure}")
    print(f"\nParametri di training:")
    print(f"  eta (learning rate): {best_params.get('eta', 0.1)}")
    print(f"  batch_size: {best_params.get('batch_size', 8)}")
    print(f"  algorithm: {best_params.get('algorithm', 'sgd')}")
    print(f"  activation_type: {best_params.get('activation_type', 'sigmoid')}")
    print(f"  loss_type: {best_params.get('loss_type', 'half_mse')}")
    print(f"  epochs: {best_params.get('epochs', 2000)}")
    print(f"  patience: {best_params.get('patience', 100)}")
    
    if 'l2_lambda' in best_params and best_params['l2_lambda'] > 0:
        print(f"  l2_lambda (regolarizzazione): {best_params['l2_lambda']}")
    
    if 'momentum' in best_params and best_params['momentum'] > 0:
        print(f"  momentum: {best_params['momentum']}")

if __name__ == "__main__":
    main()