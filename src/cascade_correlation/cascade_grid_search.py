import itertools
import numpy as np
import matplotlib.pyplot as plt
from cascade_correlation import CascadeNetwork
from data_manipulation import *
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
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
            
            # Salvataggio min/max come attributi di classe per usarli dop
            self.x_min = self.X_train.min(axis=0)
            self.x_max = self.X_train.max(axis=0)
            self.y_min = self.y_train.min(axis=0)
            self.y_max = self.y_train.max(axis=0)
            
            # Normalizzazione (Training su dati normalizzati tra -1 e 1)
            self.X_train = normalize(self.X_train, -1, 1, self.x_min, self.x_max)
            self.X_val = normalize(self.X_val, -1, 1, self.x_min, self.x_max)
            self.X_test = normalize(self.X_test, -1, 1, self.x_min, self.x_max)
            
            self.y_train = normalize(self.y_train, -1, 1, self.y_min, self.y_max)
            self.y_val = normalize(self.y_val, -1, 1, self.y_min, self.y_max)
            self.y_test = normalize(self.y_test, -1, 1, self.y_min, self.y_max)
        
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
                algorithm=params['algorithm'],
                l2_lambda=params.get('l2_lambda', 0.0)
            )
            
            final_error = net.train(
                self.X_train, self.y_train, 
                X_val = self.X_val, y_val = self.y_val,
                max_epochs=params.get('epochs', 1000), 
                tolerance=params['tolerance'], 
                patience=params['patience'], 
                max_hidden_units=params['max_hidden_units']
            )
            
            loss_history = net.loss_history
            val_loss_history = getattr(net, 'val_loss_history', [])
            
            y_pred_train = np.array([net.forward(x) for x in self.X_train])
            y_pred_val = np.array([net.forward(x) for x in self.X_val])
            y_pred_test = np.array([net.forward(x) for x in self.X_test])
            
            # Calcolo Loss (MSE su dati normalizzati)
            train_loss = np.mean((self.y_train - y_pred_train)**2)
            val_loss = np.mean((self.y_val - y_pred_val)**2)
            test_loss = np.mean((self.y_test - y_pred_test)**2)
            
            results = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'min_val_loss': val_loss, 
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
                # CALCOLO MEE SU SCALA ORIGINALE 
                target_min = -1
                target_max = 1
                
                # Denormalizzazione Target
                y_train_denorm = denormalize(self.y_train, target_min, target_max, self.y_min, self.y_max)
                y_val_denorm = denormalize(self.y_val, target_min, target_max, self.y_min, self.y_max)
                y_test_denorm = denormalize(self.y_test, target_min, target_max, self.y_min, self.y_max)
                
                # Denormalizzazione Predizioni
                pred_train_denorm = denormalize(y_pred_train, target_min, target_max, self.y_min, self.y_max)
                pred_val_denorm = denormalize(y_pred_val, target_min, target_max, self.y_min, self.y_max)
                pred_test_denorm = denormalize(y_pred_test, target_min, target_max, self.y_min, self.y_max)
                
                # Calcolo MEE reale
                train_mee = net.compute_mee(y_train_denorm, pred_train_denorm)
                val_mee = net.compute_mee(y_val_denorm, pred_val_denorm)
                test_mee = net.compute_mee(y_test_denorm, pred_test_denorm)
                
                results['train_mee'] = train_mee
                results['val_mee'] = val_mee
                results['test_mee'] = test_mee
                
            return results, net
            
        except Exception as e:
            print(f"Errore: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def coarse_grid_search(self):
        print("INIZIO COARSE SEARCH CASCADE")

        common_fixed = {'epochs': [2000], 'algorithm': ['quickprop', 'rprop']}
        if self.dataset_name == 'cup':
            param_grid = {
                'learning_rate': [0.001, 0.005, 0.01, 0.05],
                'patience': [50, 80],
                'tolerance': [0.001, 0.0001],
                'max_hidden_units': [20],  # Fisso e più alto
                'algorithm': ['quickprop', 'rprop'],
                **common_fixed
            }
            
        """if self.dataset_name in ['monk1', 'monk2']:
             param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'patience': [30, 50],
                'tolerance': [0.01, 0.001],
                'max_hidden_units': [12],  # Fisso
                'algorithm': ['quickprop', 'rprop'],
                **common_fixed
            }
        else: # monk3
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'patience': [30],            # Meno pazienza, fermati prima
                'tolerance': [0.01],         # Tolleranza più alta per ignorare il rumore
                'max_hidden_units': [3, 5],  # Pochi neuroni! Max 3-5 sono spesso sufficienti
                'l2_lambda': [1e-4, 1e-3, 1e-2], # REGOLARIZZAZIONE FONDAMENTALE
                **common_fixed
            } """
        
        all_combinations = []
        
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            
            if params['algorithm'] == 'sgd' and params.get('momentum', 0.0) == 0.0:
                params['mu'] = 1.75
                all_combinations.append(params)
            elif params['algorithm'] in ['rprop', 'quickprop']:
                params['momentum'] = 0.0
                all_combinations.append(params)
        
        print(f"Numero totale di combinazioni: {len(all_combinations)}")
  
        num_cores = multiprocessing.cpu_count()
        print(f"Avvio esecuzione parallela su {num_cores} core...")
        
        parallel_results = Parallel(n_jobs=-1)(
            delayed(self.train_evaluate)(params) 
            for params in tqdm(all_combinations, desc="Coarse Grid Search (Parallel)")
        )
        
        best_score = float('inf')
        best_params = None
        all_results = []
        
        # Riaggregazione dei risultati dopo il parallelo
        for i, (results, _) in enumerate(parallel_results):
            params = all_combinations[i]
            
            if results is not None:
                all_results.append((params, results))
                
                # Usa validation loss (MSE normalizzato) per la selezione del modello migliore
                current_score = results['min_val_loss']
                
                if current_score < best_score:
                    best_score = current_score
                    best_params = params.copy()
        
        print("\n" + "="*80)
        print("MIGLIORI RISULTATI COARSE GRID SEARCH")
        print("="*80)
        
        all_results.sort(key=lambda x: x[1]['min_val_loss'])
        top_n = min(5, len(all_results))
        
        for i, (params, results) in enumerate(all_results[:top_n]):
            print(f"\nPosizione {i+1}:")
            print(f"Validation Loss (MSE Norm): {results['min_val_loss']:.6f}")
            if self.dataset_name == 'cup':
                 print(f"Validation MEE (Original):  {results['val_mee']:.6f}")
            elif self.dataset_name in ['monk1', 'monk2', 'monk3']:
                print(f"Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Parametri: {params}")
        
        return best_params, all_results[:top_n]

    def fine_grid_search(self, best_params, top_results):
        print("INIZIO FINE SEARCH CASCADE")
        
        fine_tuning_grid = {
            'learning_rate': [best_params['learning_rate']*0.8, best_params['learning_rate'], best_params['learning_rate']*1.2],
            'patience': [best_params['patience'] - 10, best_params['patience'], best_params['patience'] + 10],
            'tolerance': [best_params['tolerance']], 
            'max_hidden_units': [best_params['max_hidden_units']] 
        }
        
        if best_params['algorithm'] == 'rprop':
            ep = best_params.get('eta_plus', 1.2)
            em = best_params.get('eta_minus', 0.5)
            fine_tuning_grid['eta_plus'] = [ep - 0.1, ep, ep + 0.1]
            fine_tuning_grid['eta_minus'] = [em - 0.1, em, em + 0.1]
            
        elif best_params['algorithm'] == 'quickprop':
            current_mu = best_params.get('mu', 1.75)
            fine_tuning_grid['mu'] = [current_mu - 0.25, current_mu, current_mu + 0.25]   
        
        fixed_params = {
            'epochs': best_params['epochs'],
            'algorithm': best_params['algorithm'],
            'patience': best_params['patience'],
            'tolerance': best_params['tolerance'],
            'momentum': best_params.get('momentum', 0.0),
        }
        
        all_combinations = []
        keys = list(fine_tuning_grid.keys())
        values = list(fine_tuning_grid.values())
        
        for combo in itertools.product(*values):
            params = fixed_params.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            all_combinations.append(params)
        
        print(f"Numero di combinazioni di raffinamento: {len(all_combinations)}")
        
        # --- MODIFICA JOBLIB: Esecuzione parallela ---
        print(f"Avvio esecuzione parallela fine search...")
        
        parallel_results = Parallel(n_jobs=-1)(
            delayed(self.train_evaluate)(params) 
            for params in tqdm(all_combinations, desc="Fine Grid Search (Parallel)")
        )
        # ---------------------------------------------
        
        best_fine_score = float('inf')
        best_fine_params = None
        fine_results = []
        
        # Riaggregazione
        for i, (results, _) in enumerate(parallel_results):
            params = all_combinations[i]
            
            if results is not None:
                fine_results.append((params, results))
                current_score = results['min_val_loss']
                
                if current_score < best_fine_score:
                    best_fine_score = current_score
                    best_fine_params = params.copy()
        
        print("\n" + "="*80)
        print("MIGLIORI RISULTATI FINE GRID SEARCH")
        print("="*80)
        
        fine_results.sort(key=lambda x: x[1]['min_val_loss'])
        
        for i, (params, results) in enumerate(fine_results[:3]):
            print(f"\nPosizione {i+1}:")
            print(f"Validation Loss (MSE Norm): {results['min_val_loss']:.6f}")
            if self.dataset_name == 'cup':
                 print(f"Validation MEE (Original):  {results['val_mee']:.6f}")
            elif self.dataset_name in ['monk1', 'monk2', 'monk3']:
                print(f"Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Parametri: {params}")
        
        return best_fine_params, fine_results[0]
    
    def plot_results(self, history, best_params, results):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        training_loss = []
        validation_loss = []
            
        if isinstance(history, (list, np.ndarray)):
            training_loss = history
        
        # Plot Loss
        axes[0].plot(training_loss, label='Training Loss', alpha=0.7)
        if 'val_history' in results and len(results['val_history']) > 0:
            axes[0].plot(results['val_history'], label='Validation Loss', linestyle='--', color='orange', alpha=0.7)
        axes[0].set_xlabel('Epoche')
        axes[0].set_ylabel('Loss (MSE Normalized)')
        axes[0].set_title(f'Learning Curve - {self.dataset_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log') # Log scale per vedere meglio la discesa
        
        # Bar Plot Metriche
        if self.dataset_name in ['monk1', 'monk2', 'monk3']:
            metrics = ['Train Acc', 'Val Acc', 'Test Acc']
            values = [results.get('train_accuracy', 0), results.get('val_accuracy', 0), results.get('test_accuracy', 0)]
            title = 'Accuracy Finale'
            ylim = [0, 1.1]
        else:
            # Per CUP mostriamo il MEE
            metrics = ['Train MEE', 'Val MEE', 'Test MEE']
            values = [results.get('train_mee', 0), results.get('val_mee', 0), results.get('test_mee', 0)]
            title = 'MEE Finale (Original Scale)'
            ylim = None
            
        bars = axes[1].bar(metrics, values, color=['blue', 'green', 'red'])
        axes[1].set_title(title)
        if ylim: axes[1].set_ylim(ylim)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        filename = f'cascade_results_{self.dataset_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nGrafico salvato come: {filename}")
    
    def run(self):
        print(f"\n{'='*80}")
        print(f"GRID SEARCH PER DATASET: {self.dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        best_coarse_params, top_results = self.coarse_grid_search()
        
        if not best_coarse_params:
            print("Errore: nessun risultato valido dalla grid search")
            return
        
        best_fine_params, (final_params, final_results) = self.fine_grid_search(best_coarse_params, top_results)
        
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
            else:
                print(f"  Training Loss (MSE Norm): {final_results['train_loss']:.6f}")
                print(f"  Training MEE (Original):  {final_results['train_mee']:.6f}")
                print(f"  Validation MEE (Original):{final_results['val_mee']:.6f}")
                print(f"  Test MEE (Original):      {final_results['test_mee']:.6f}")
            
            self.plot_results(final_results['history'], final_params, final_results)
        
        return final_params, final_results, final_model

def main():
    # Lista di tutti i dataset su cui vuoi fare la grid search
    datasets = ['monk1', 'monk2', 'monk3', 'cup']
    
    for dataset in datasets:
        print(f"\n\n{'#'*80}")
        print(f"ESECUZIONE PER DATASET: {dataset}")
        print(f"{'#'*80}")
        try:
            grid_search = CascadeGridSearch(dataset_name=dataset)
            grid_search.run()
        except Exception as e:
            print(f"Errore durante l'esecuzione su {dataset}: {e}")
            continue # Passa al prossimo dataset anche se uno fallisce

if __name__ == "__main__":
    main()