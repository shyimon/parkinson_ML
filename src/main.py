import argparse
import numpy as np
import data_manipulation as data
import new_grid_search 

def main():
    parser = argparse.ArgumentParser(description='Grid Search per Neural Network')
    parser.add_argument('--dataset', type=str, default='monk3',
                       choices=['monk1', 'monk2', 'monk3', 'cup'],
                       help='Dataset da usare')
    parser.add_argument('--no_one_hot', action='store_true',
                       help='Disabilita one-hot encoding (solo per Monk)')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='Disabilita shuffle dei dati')
    parser.add_argument('--train_size', type=int, default=250,
                       help='Training size per CUP')
    parser.add_argument('--val_size', type=int, default=125,
                       help='Validation size per CUP')
    parser.add_argument('--test_size', type=int, default=125,
                       help='Test size per CUP')
    parser.add_argument('--quick_test', action='store_true',
                       help='Esegui un test rapido con meno trials')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GRID SEARCH AVANZATO PER NEURAL NETWORK")
    print("=" * 70)
    
    # Esegui il grid search
    try:
        if args.quick_test:
            # Per test rapido, modifica il file new_grid_search.py per usare meno trials
            print("Modalità test rapido attivata...")
            # Qui potresti importare una versione modificata o passare parametri
            # Per semplicità, eseguiamo comunque il grid search normale
            pass
        
        results = new_grid_search.run_grid_search(
            dataset_name=args.dataset,
            one_hot=not args.no_one_hot,
            dataset_shuffle=not args.no_shuffle,
            cup_train_size=args.train_size,
            cup_val_size=args.val_size,
            cup_test_size=args.test_size
        )
        
        # Stampa un riepilogo
        print("\n" + "=" * 70)
        print("RIEPILOGO RISULTATI")
        print("=" * 70)
        
        best_params = results['best_params']
        network_structure = results['network_structure']
        
        print(f"\nSTRUTTURA RETE OTTIMALE:")
        print(f"  {network_structure}")
        print(f"  Input: {network_structure[0]} neuroni")
        
        hidden_layers = network_structure[1:-1]
        for i, neurons in enumerate(hidden_layers):
            print(f"  Hidden layer {i+1}: {neurons} neuroni")
        
        print(f"  Output: {network_structure[-1]} neurone/i")
        
        print(f"\nIPERPARAMETRI OTTIMALI:")
        for key in ['eta', 'batch_size', 'epochs', 'algorithm', 
                   'activation_type', 'loss_type', 'l2_lambda', 'momentum']:
            if key in best_params:
                print(f"  {key}: {best_params[key]}")
        
        if 'hidden_structure' in best_params:
            print(f"  hidden_structure: {best_params['hidden_structure']}")
        
        print(f"\nPERFORMANCE FINALE (300 trials):")
        if args.dataset == 'cup':
            print(f"  Training MEE: {results['train_mean']:.4f} ± {results['train_std']:.4f}")
            print(f"  Validation MEE: {results['val_mean']:.4f} ± {results['val_std']:.4f}")
            print(f"  Test MEE: {results['test_mean']:.4f} ± {results['test_std']:.4f}")
        else:
            print(f"  Training Accuracy: {results['train_mean']:.2f}% ± {results['train_std']:.2f}%")
            print(f"  Validation Accuracy: {results['val_mean']:.2f}% ± {results['val_std']:.2f}%")
            print(f"  Test Accuracy: {results['test_mean']:.2f}% ± {results['test_std']:.2f}%")
        
    except Exception as e:
        print(f"\nErrore durante il grid search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
COME ESEGUIRE
# Monk3 (default)
python main.py

# Monk1
python main.py --dataset monk1

# Monk2
python main.py --dataset monk2

# CUP
python main.py --dataset cup

# Disabilitare one-hot encoding
python main.py --dataset monk3 --no_one_hot

# Disabilitare shuffle
python main.py --dataset monk3 --no_shuffle

# CUP con dimensioni personalizzate
python main.py --dataset cup --train_size 300 --val_size 100 --test_size 100
"""