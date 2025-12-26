import itertools
import numpy as np
import matplotlib.pyplot as plt
from cascade_correlation import CascadeNetwork
from data_manipulation import *
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CascadeGridSearch:
    def __init__(self, dataset_name='monk1'):
        self.dataset_name = dataset_name
        self.load_dataset() 
        
    def load_dataset(self):
        if self.dataset_name == 'monk1':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = return_monk1(dataset_shuffle=True, one_hot=True)
        elif self.dataset_name == 'monk2':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = return_monk2(dataset_shuffle=True, one_hot=True)
        elif self.dataset_name == 'monk3':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = return_monk3(dataset_shuffle=True, one_hot=True, val_split=0.3)
        elif self.dataset_name == 'cup':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = return_CUP(dataset_shuffle=True, train_size=250, validation_size=125, test_size=125)
            
            # Salva minimi e massimi per denormalizzazione futura
            self.x_min = self.X_train.min(axis=0)
            self.x_max = self.X_train.max(axis=0)
            self.y_min = self.y_train.min(axis=0)
            self.y_max = self.y_train.max(axis=0)
            # Normalizzazione per CUP
            x_min = self.X_train.min(axis=0)
            x_max = self.X_train.max(axis=0)
            y_min = self.y_train.min(axis=0)
            y_max = self.y_train.max(axis=0)
            
            self.X_train = normalize(self.X_train, -1, 1, x_min, x_max)
            self.X_val = normalize(self.X_val, -1, 1, x_min, x_max)
            self.X_test = normalize(self.X_test, -1, 1, x_min, x_max)
            
            self.y_train = normalize(self.y_train, -1, 1, y_min, y_max)
            self.y_val = normalize(self.y_val, -1, 1, y_min, y_max)
            self.y_test = normalize(self.y_test, -1, 1, y_min, y_max)
        
        print(f"Dataset {self.dataset_name} caricato:")
        print(f"  Training: {self.X_train.shape[0]} esempi, {self.X_train.shape[1]} features")
        print(f"  Validation: {self.X_val.shape[0]} esempi")
        print(f"  Test: {self.X_test.shape[0]} esempi")
        
    def train_evaluate(self, params, return_history=False):
        try:
            input_dim = self.X_train.shape[1]
            output_dim = self.y_train.shape[1]

            net = CascadeNetwork(
                input_dim, 
                output_dim, 
                learning_rate=params['learning_rate'], 
                algorithm=params['algorithm']
            )

            _ = net.train(
                self.X_train, self.y_train, 
                X_val = self.X_val, y_val = self.y_val,
                max_epochs=params.get('epochs', 1000), 
                tolerance=params['tolerance'], 
                patience=params['patience'], 
                max_hidden_units=params['max_hidden_units']
            )
            
            loss_history = net.loss_history
            val_loss_history = net.val_loss_history
            
            y_pred_train = np.array([net.forward(x) for x in self.X_train])
            y_pred_val = np.array([net.forward(x) for x in self.X_val])
            y_pred_test = np.array([net.forward(x) for x in self.X_test])
            
            train_loss = np.mean((self.y_train - y_pred_train)**2)
            val_loss = np.mean((self.y_val - y_pred_val)**2)
            test_loss = np.mean((self.y_test - y_pred_test)**2)
            
            results = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'min_val_loss': np.min(net.val_loss_history) if len(net.val_loss_history) > 0 else val_loss, # minimo storico di validation loss 
                'history': loss_history,
                'val_history': val_loss_history
            }
            
            if self.dataset_name != 'cup':
                train_accuracy = np.mean((y_pred_train > 0.5).astype(int) == self.y_train)
                results['train_accuracy'] = train_accuracy
                val_accuracy = np.mean((y_pred_val > 0.5).astype(int) == self.y_val)
                results['val_accuracy'] = val_accuracy
                test_accuracy = np.mean((y_pred_test > 0.5).astype(int) == self.y_test)
                results['test_accuracy'] = test_accuracy
            else:
                target_min = -1
                target_max = 1
                
                # Training
                y_train_denorm = denormalize(self.y_train, target_min, target_max, self.y_min, self.y_max)
                pred_train_denorm = denormalize(y_pred_train, target_min, target_max, self.y_min, self.y_max)
                
                # Validation
                y_val_denorm = denormalize(self.y_val, target_min, target_max, self.y_min, self.y_max)
                pred_val_denorm = denormalize(y_pred_val, target_min, target_max, self.y_min, self.y_max)
                
                # Test
                y_test_denorm = denormalize(self.y_test, target_min, target_max, self.y_min, self.y_max)
                pred_test_denorm = denormalize(y_pred_test, target_min, target_max, self.y_min, self.y_max)
                
                # 2. Calcolo MEE su dati denormalizzati
                train_mee = net.compute_mee(y_train_denorm, pred_train_denorm)
                val_mee = net.compute_mee(y_val_denorm, pred_val_denorm)
                test_mee = net.compute_mee(y_test_denorm, pred_test_denorm)
                
                # 3. Salva nei risultati
                results['train_mee'] = train_mee
                results['val_mee'] = val_mee
                results['test_mee'] = test_mee
                
            return results, net
            
        except Exception as e:
            print(f"Errore: {e}")
            return None, None

    def coarse_grid_search(self):
        print("INIZIO GRID SEARCH")

        # PARAMETRI FISSI PER TUTTE LE COMBINAZIONI
        fixed_max_units = 10 if 'monk' in self.dataset_name else 25 
        fixed_patience = 40
        fixed_tolerance = 1e-4 # Molto bassa per forzare l'early stopping
        fixed_epochs = 1000 

        # DEFINIZIONE GRIGLIA SPECIFICA PER DATASET
        if self.dataset_name == 'cup':
            # CUP: Problema di regressione complesso. 
            # Serve regolarizzazione (L2) e learning rate medio-bassi.
            param_grid = {
                'learning_rate': [0.001, 0.005, 0.01],  
                'l2_lambda': [1e-4, 1e-3, 1e-2],       
                'algorithm': ['sgd', 'quickprop'],      
                'momentum': [0.5, 0.9] if 'sgd' in ['sgd'] else [0.0], # Momentum solo se usi SGD vanilla
                'mu': [1.75], 
                'eta_plus': [1.2],
                'eta_minus': [0.5]
            }
        
        elif self.dataset_name in ['monk1', 'monk2']:
            # MONK1/2
            param_grid = {
                'learning_rate': [0.005, 0.01],   
                'l2_lambda': [0.0, 1e-6],               
                'algorithm': ['quickprop', 'rprop'],    
                'mu': [1.75],       
                'eta_plus': [1.2],  
                'eta_minus': [0.5], 
                'momentum': [0.0],
                'max_hidden_units': [1, 2],
                'patience': [20, 25]
            }
            
        else: 
            # MONK3: Ha rumore (5%), serve un minimo di regolarizzazione
            param_grid = {
                'learning_rate': [0.005, 0.01],
                'l2_lambda': [1e-4, 1e-3], 
                'algorithm': ['quickprop'],
                'mu': [1.75],
                'eta_plus': [1.2],
                'eta_minus': [0.5],
                'momentum': [0.0]
            }

        all_combinations = []
        import itertools
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            
            # Aggiungi i parametri fissi che abbiamo tolto dalla griglia
            # params['patience'] = fixed_patience
            params['tolerance'] = fixed_tolerance
            # params['max_hidden_units'] = fixed_max_units
            params['epochs'] = fixed_epochs

            # Regole di esclusione per evitare combinazioni inutili
            if params['algorithm'] == 'sgd':
                # SGD non usa mu, eta_plus, eta_minus
                if 'mu' in params: del params['mu'] 
                # SGD ha bisogno di momentum, gli altri no
            elif params['algorithm'] == 'quickprop':
                if params.get('momentum', 0) > 0: continue 
                params['eta_plus'] = 0 # Non usati
                params['eta_minus'] = 0
            elif params['algorithm'] == 'rprop':
                if params.get('momentum', 0) > 0: continue
                params['mu'] = 0

            if params not in all_combinations:
                all_combinations.append(params)

        print(f"Numero totale di combinazioni ottimizzate: {len(all_combinations)}")
        
        best_score = float('inf')
        best_params = None
        all_results = []
        
        for i, params in enumerate(tqdm(all_combinations, desc="Smart Grid Search")):
            results, _ = self.train_evaluate(params)
            
            if results is not None:
                all_results.append((params, results))
                
                # Per MONK usiamo Accuracy, per CUP usiamo Loss
                if self.dataset_name == 'cup':
                    current_score = results['min_val_loss']
                else:
                    current_score = results['min_val_loss'] 
                
                if current_score < best_score:
                    best_score = current_score
                    best_params = params.copy()

        return best_params, all_results[:5]

    def fine_grid_search(self, best_params, top_results):
        print("INIZIO FINE SEARCH CASCADE")
    
        fine_tuning_grid = {
            'learning_rate': [best_params['learning_rate']*0.8, best_params['learning_rate'], best_params['learning_rate']*1.2],
            'patience': [best_params['patience'] - 10, best_params['patience'], best_params['patience'] + 10],
            'tolerance': [best_params['tolerance']], 
            'max_hidden_units': [best_params['max_hidden_units']] 
        }
        
        # Aggiungi parametri specifici dell'algoritmo
        if best_params['algorithm'] == 'rprop':
            ep = best_params.get('eta_plus', 1.2)
            em = best_params.get('eta_minus', 0.5)
            
            fine_tuning_grid['eta_plus'] = [ep - 0.1, ep, ep + 0.1]
            fine_tuning_grid['eta_minus'] = [em - 0.1, em, em + 0.1]
            
        elif best_params['algorithm'] == 'quickprop':
            current_mu = best_params.get('mu', 1.75)
            fine_tuning_grid['mu'] = [current_mu - 0.25, current_mu, current_mu + 0.25]   
        
        # Mantieni gli altri parametri fissi
        fixed_params = {
            'epochs': best_params['epochs'],
            'algorithm': best_params['algorithm'],
            'patience': best_params['patience'],
            'tolerance': best_params['tolerance'],
            'momentum': best_params.get('momentum', 0.0),
        }
        
        # Genera tutte le combinazioni
        all_combinations = []
        keys = list(fine_tuning_grid.keys())
        values = list(fine_tuning_grid.values())
        
        for combo in itertools.product(*values):
            params = fixed_params.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            all_combinations.append(params)
        
        print(f"Numero di combinazioni di raffinamento: {len(all_combinations)}")
        
        best_fine_score = float('inf')
        best_fine_params = None
        fine_results = []
        
        # Esegui la grid search di raffinamento
        for params in tqdm(all_combinations, desc="Fine Grid Search"):
            results, _ = self.train_evaluate(params)
            
            if results is not None:
                fine_results.append((params, results))
                current_score = results['min_val_loss']
                
                if current_score < best_fine_score:
                    best_fine_score = current_score
                    best_fine_params = params.copy()
        
        print("\n" + "="*80)
        print("MIGLIORI RISULTATI FINE GRID SEARCH")
        print("="*80)
        
        # Ordina i risultati
        fine_results.sort(key=lambda x: x[1]['min_val_loss'])
        
        for i, (params, results) in enumerate(fine_results[:3]):
            print(f"\nPosizione {i+1}:")
            print(f"Validation Loss: {results['min_val_loss']:.6f}")
            if self.dataset_name in ['monk1', 'monk2', 'monk3']:
                print(f"Test Accuracy: {results['test_accuracy']:.4f}")
                print(f"Train Accuracy: {results['train_accuracy']:.4f}")
            print(f"Parametri: {params}")
        
        return best_fine_params, fine_results[0]
    
    def plot_results(self, history, params, final_results):
        """
        Genera i grafici appropriati in base al dataset (MONK vs CUP).
        """
        # Creiamo una cartella per i plot se non esiste
        import os
        if not os.path.exists('plots_cascade'):
            os.makedirs('plots_cascade')
            
        # Nome base per i file
        filename_base = f"plots_cascade/{self.dataset_name}_{params['algorithm']}_eta{params['learning_rate']}_l2{params.get('l2_lambda', 0)}"
        
        # GRAFICO LOSS (valido per entrambi i dataset)
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2, linestyle='--')
        
        plt.title(f'Learning Curve (Loss) - {self.dataset_name.upper()}\nParams: {params}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log') 
        plt.savefig(f"{filename_base}_loss.png")
        plt.close()
        
        # GRAFICI SPECIFICI PER DATASET
        if self.dataset_name in ['monk1', 'monk2', 'monk3']:
            # PER MONK: Stampiamo l'ACCURACY
            if 'train_acc' in history and 'val_acc' in history:
                plt.figure(figsize=(10, 6))
                plt.plot(history['train_acc'], label='Training Accuracy', color='green')
                plt.plot(history['val_acc'], label='Validation Accuracy', color='red', linestyle='--')
                
                plt.title(f'Accuracy Curve - {self.dataset_name.upper()}')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                plt.ylim(0, 1.05)
                plt.savefig(f"{filename_base}_accuracy.png")
                plt.close()
                
        else:
            # PER CUP: Stampiamo MEE e SCATTER PLOT
            if 'train_mee' in history and 'val_mee' in history:
                plt.figure(figsize=(10, 6))
                plt.plot(history['train_mee'], label='Training MEE', color='purple')
                plt.plot(history['val_mee'], label='Validation MEE', color='brown', linestyle='--')
                plt.title(f'MEE Curve - CUP')
                plt.xlabel('Epochs')
                plt.ylabel('Mean Euclidean Error')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{filename_base}_mee.png")
                plt.close()
                
            pass
        
    def run(self):
        #esegue la grid search completa
        print(f"\n{'='*80}")
        print(f"GRID SEARCH PER DATASET: {self.dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Grid search grossolana
        best_coarse_params, top_results = self.coarse_grid_search()
        
        if not best_coarse_params:
            print("Errore: nessun risultato valido dalla grid search")
            return
        
        # Grid search con raffinamento
        best_fine_params, (final_params, final_results) = self.fine_grid_search(best_coarse_params, top_results)
        
        #  Addestra il modello finale con i migliori parametri
        print("\n" + "="*80)
        print("ADDESTRAMENTO FINALE CON I MIGLIORI PARAMETRI")
        print("="*80)
        
        final_results, final_model = self.train_evaluate(final_params, return_history=True)
        
        if final_results and final_model:
            print("\nRISULTATI FINALI:")
            print(f"Parametri finali: {final_params}")
            print(f"\nPerformance:")
            
            if self.dataset_name in ['monk1', 'monk2', 'monk3']:
                print(f"  Training Accuracy:   {final_results['train_accuracy']:.4f}")
                print(f"  Validation Accuracy: {final_results['val_accuracy']:.4f}")
                print(f"  Test Accuracy:       {final_results['test_accuracy']:.4f}")
                print(f"  Training Loss:       {final_results['train_loss']:.6f}")
                print(f"  Validation Loss:     {final_results['val_loss']:.6f}")
                print(f"  Test Loss:           {final_results['test_loss']:.6f}")
                print(f"  Min Validation Loss: {final_results['min_val_loss']:.6f}")
            else:
                print(f"  Training Loss (MSE): {final_results['train_loss']:.6f}")
                print(f"  Validation MEE:      {final_results.get('val_mee', 0):.6f}") 
                print(f"  Test MEE:            {final_results.get('test_mee', 0):.6f}") 
                print(f"  Training MEE:        {final_results.get('train_mee', 0):.6f}")
            
            # Crea e mostra i grafici
            self.plot_results(final_results['history'], final_params, final_results)
        
        return final_params, final_results, final_model


def main():
    # Scegli il dataset da usare
    # Opzioni: 'monk1', 'monk2', 'monk3', 'cup'
    # dataset_name = 'monk1'  # Cambia questo per testare dataset diversi
    
    # Crea e esegui la grid search
    datasets = ['monk1', 'monk2', 'monk3', 'cup']
    for dataset in datasets:
        print(f"\n\n{'#'*80}")
        print(f"ESECUZIONE PER DATASET: {dataset}")
        print(f"{'#'*80}")
        grid_search = CascadeGridSearch(dataset_name=dataset)
        grid_search.run()


if __name__ == "__main__":
    main()