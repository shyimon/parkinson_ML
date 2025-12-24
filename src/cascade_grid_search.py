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
            
            return results, net
            
        except Exception as e:
            print(f"Errore: {e}")
            return None, None

    def coarse_grid_search(self):
        print("INIZIO COARSE SEARCH CASCADE")

        if self.dataset_name == 'cup':
            
            param_grid = {
                'learning_rate': [0.005, 0.01, 0.05],   
                'patience': [30, 50, 70],                 
                'tolerance': [0.001, 0.005, 0.01],           
                'max_hidden_units': [5, 10, 15],  # Importante per limitare la crescita    
                'algorithm': ['sgd', 'quickprop', 'rprop'],
                'l2_lambda': [0.0001, 0.001, 0.01],  # Aggiunta regolarizzazione L2
                'epochs': [2000], # Solitamente fisso alto, tanto si ferma con la pazienza
                'momentum': [0.0],  
                'mu': [0.0],
                'eta_plus': [0.0],
                'eta_minus': [0.0]
            }
        else:
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],   
                'patience': [20, 30],                 
                'tolerance': [0.01, 0.02, 0.05],           
                'max_hidden_units': [1, 2],  # Importante per limitare la crescita    
                'algorithm': ['rprop', 'quickprop'],
                'momentum': [0.0],  
                'mu': [1.25, 1.5, 1.75],      # Solo per Quickprop
                'eta_plus': [1.1, 1.2, 1.3],  # Solo per RPROP
                'eta_minus': [0.4, 0.5, 0.6], # Solo per RPROP
                'epochs': [1000], # Solitamente fisso alto, tanto si ferma con la pazienza
                'l2_lambda': [0.01, 0.05, 0.1]
            }
        
        # Filtra i parametri non necessari per certi algoritmi
        all_combinations = []
        default_mu = param_grid['mu'][0]
        default_ep = param_grid['eta_plus'][0]
        default_em = param_grid['eta_minus'][0]
        
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            
            algo = params['algorithm']
            # Filtra combinazioni inappropriate
            if algo == 'sgd':
                if params['mu'] != default_mu or params['eta_plus'] != default_ep or params['eta_minus'] != default_em:
                    continue
                all_combinations.append(params)
            elif algo == 'rprop':
                if params['momentum'] != 0.0 or params['mu'] != default_mu:
                    continue
                all_combinations.append(params)
            elif algo == 'quickprop':
                if params['momentum'] != 0.0 or params['eta_plus'] != default_ep or params['eta_minus'] != default_em:
                    continue
            all_combinations.append(params)
                
        print(f"Numero totale di combinazioni: {len(all_combinations)}")
        
        best_score = float('inf')
        best_params = None
        best_history = None
        all_results = []
        
        # Esegui la grid search
        for i, params in enumerate(tqdm(all_combinations, desc="Grid Search")):
            results, _ = self.train_evaluate(params)
            
            if results is not None:
                all_results.append((params, results))
                
                # Usa validation loss come metrica principale
                current_score = results['min_val_loss']
                
                if current_score < best_score:
                    best_score = current_score
                    best_params = params.copy()
                    print(f"\nNuovo miglior risultato: {current_score:.6f}")
                    print(f"Parametri: {params}")
                    
                    if self.dataset_name in ['monk1', 'monk2', 'monk3']:
                        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
                        print(f"Train Accuracy: {results['train_accuracy']:.4f}")
        
        print("\n" + "="*80)
        print("MIGLIORI RISULTATI COARSE GRID SEARCH")
        print("="*80)
        
        # Trova i migliori risultati
        all_results.sort(key=lambda x: x[1]['min_val_loss'])
        top_n = min(5, len(all_results))
        
        for i, (params, results) in enumerate(all_results[:top_n]):
            print(f"\nPosizione {i+1}:")
            print(f"Validation Loss: {results['min_val_loss']:.6f}")
            if self.dataset_name in ['monk1', 'monk2', 'monk3']:
                print(f"Test Accuracy: {results['test_accuracy']:.4f}")
                print(f"Train Accuracy: {results['train_accuracy']:.4f}")
            print(f"Parametri: {params}")
        
        return best_params, all_results[:top_n]

    def fine_grid_search(self, best_params, top_results):
        print("INIZIO FINE SEARCH CASCADE")
        
        # ADATTAMENTO: Raffinamento specifico
        # Esempio: cerchiamo un learning rate più preciso intorno al vincitore
        fine_tuning_grid = {
            'learning_rate': [best_params['learning_rate']*0.8, best_params['learning_rate'], best_params['learning_rate']*1.2],
            'patience': [best_params['patience'] - 10, best_params['patience'], best_params['patience'] + 10],
            # Magari proviamo a cambiare leggermente la tolleranza
            'tolerance': [best_params['tolerance']], 
            'max_hidden_units': [best_params['max_hidden_units']] # Manteniamo fisso o variamo di poco
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
    
    def plot_results(self, history, best_params, results):
        """Crea grafici dei risultati"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. Recupera le liste corrette
        training_loss = history
        # Cerca 'val_history' nel dizionario dei risultati, se non c'è usa lista vuota
        validation_loss = results.get('val_history', []) 
        
        # Grafico 1: Loss Curves
        axes[0].plot(training_loss, label='Training Loss', alpha=0.7)
        
        # 2. Plotta la validation solo se contiene dati
        if len(validation_loss) > 0:
            axes[0].plot(validation_loss, label='Validation Loss', alpha=0.7, linestyle='--')
            
            # Minima validation loss
            min_val_idx = np.argmin(validation_loss)
            min_val_val = validation_loss[min_val_idx]
            axes[0].axvline(x=min_val_idx, color='r', linestyle=':', alpha=0.5)
            # Aggiunge un pallino nel punto di minimo
            axes[0].scatter(min_val_idx, min_val_val, c='red', s=30, zorder=5, label='Min Val Loss')
            
        axes[0].set_xlabel('Epoche')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Learning Curve - {self.dataset_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Grafico 2: Bar plot delle metriche finali 
        if self.dataset_name in ['monk1', 'monk2', 'monk3']:
            metrics = ['Train Acc', 'Val Acc', 'Test Acc']
            values = [results.get('train_accuracy', 0), results.get('val_accuracy', 0), results.get('test_accuracy', 0)]
            title = 'Accuracy Finale'
            ylim = [0, 1.1]
        else:
            # Per regressione (CUP)
            metrics = ['Train Loss', 'Val Loss', 'Test Loss']
            values = [results['train_loss'], results['val_loss'], results['test_loss']]
            title = 'MSE Finale'
            ylim = None
            
        bars = axes[1].bar(metrics, values, color=['blue', 'green', 'red'])
        axes[1].set_title(title)
        if ylim: axes[1].set_ylim(ylim)
        
        # Aggiungi i valori sulle barre
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Salva il grafico
        filename = f'cascade_results_{self.dataset_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        print(f"\nGrafico salvato come: {filename}")
        
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
                print(f"  Training Loss:       {final_results['train_loss']:.6f}")
                print(f"  Validation Loss:     {final_results['val_loss']:.6f}")
                print(f"  Test Loss:           {final_results['test_loss']:.6f}")
                print(f"  Min Validation Loss: {final_results['min_val_loss']:.6f}")
            
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