import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_manipulation import *
from neural_network import NeuralNetwork

class GridSearch:
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
            self.X_train, self.X_val, self.X_test = normalize_dataset(self.X_train, self.X_val, self.X_test)
        
        print(f"Dataset {self.dataset_name} caricato:")
        print(f"  Training: {self.X_train.shape[0]} esempi, {self.X_train.shape[1]} features")
        print(f"  Validation: {self.X_val.shape[0]} esempi")
        print(f"  Test: {self.X_test.shape[0]} esempi")
    
    def create_network_structure(self, hidden_layers_config):
       # Crea la struttura della rete neurale basata sul dataset e sulla configurazione dei layer nascosti
        input_dim = self.X_train.shape[1]
        
        if self.dataset_name in ['monk1', 'monk2', 'monk3']:
            output_dim = 1  # Classificazione binaria
        elif self.dataset_name == 'cup':
            output_dim = self.y_train.shape[1]  # Regressione multi-target
        
        structure = [input_dim]
        for layer_config in hidden_layers_config:
            if isinstance(layer_config, tuple):
                structure.append(layer_config[0])  # Numero di neuroni
            else:
                structure.append(layer_config)
        structure.append(output_dim)
        
        return structure
    
    def train_evaluate(self, params, return_history=False):
        # Esegue l'addestramento e la valutazione con i parametri specificati
        try:
            # Crea struttura della rete
            structure = self.create_network_structure(params['hidden_layers'])
            
            # Crea la rete neurale
            nn = NeuralNetwork(
                structure,
                eta=params['eta'],
                loss_type=params['loss_type'],
                l2_lambda=params['l2_lambda'],
                algorithm=params['algorithm'],
                activation_type=params['activation_type'],
                weight_initializer=params['weight_initializer'],
                momentum=params.get('momentum', 0.0),
                mu=params.get('mu', 1.75),
                decay=params.get('decay', 0.9),
                eta_plus=params.get('eta_plus', 1.2),
                eta_minus=params.get('eta_minus', 0.5),
                debug=False
            )
            
            # Addestra il modello
            history = nn.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                patience=params['patience'],
                verbose=False
            )
            
            # Valutazione finale
            y_pred_train = nn.predict(self.X_train)
            y_pred_val = nn.predict(self.X_val)
            y_pred_test = nn.predict(self.X_test)
            
            # Calcola accuracy/loss in base al tipo di problema
            if self.dataset_name in ['monk1', 'monk2', 'monk3']:
                # Classificazione binaria
                train_acc = np.mean((y_pred_train > 0.5).astype(int) == self.y_train)
                val_acc = np.mean((y_pred_val > 0.5).astype(int) == self.y_val)
                test_acc = np.mean((y_pred_test > 0.5).astype(int) == self.y_test)
                
                train_loss = np.mean(nn.compute_loss(self.y_train, y_pred_train, params['loss_type']))
                val_loss = np.mean(nn.compute_loss(self.y_val, y_pred_val, params['loss_type']))
                test_loss = np.mean(nn.compute_loss(self.y_test, y_pred_test, params['loss_type']))
                
                results = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                    'min_val_loss': min(history['validation']),
                    'history': history if return_history else None
                }
            else:
                # Regressione (CUP)
                train_loss = np.mean(nn.compute_loss(self.y_train, y_pred_train, params['loss_type']))
                val_loss = np.mean(nn.compute_loss(self.y_val, y_pred_val, params['loss_type']))
                test_loss = np.mean(nn.compute_loss(self.y_test, y_pred_test, params['loss_type']))
                
                results = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                    'min_val_loss': min(history['validation']),
                    'history': history if return_history else None
                }
            
            return results, nn
            
        except Exception as e:
            print(f"Errore durante il training con parametri {params}: {str(e)}")
            return None, None
    
    def coarse_grid_search(self):
        print("="*80)
        print("GRID SEARCH")
        print("="*80)
        
        # Definisci lo spazio degli iperparametri
        param_grid = {
            'epochs': [400, 500, 600, 700],
            'eta': [0.01, 0.1, 0.15, 0.2],
            'l2_lambda': [0.0, 0.0001, 0.001],
            'algorithm': ['sgd', 'rprop', 'quickprop'],
            'activation_type': ['sigmoid', 'tanh'],
            'weight_initializer': ['def', 'xavier'],
            'loss_type': ['half_mse', 'binary_crossentropy'] if self.dataset_name in ['monk1', 'monk2', 'monk3'] else ['half_mse', 'mae'],
            'eta_plus': [1.1, 1.2, 1.3],
            'eta_minus': [0.4, 0.5, 0.6],
            'decay': [0.8, 0.9, 0.95],
            'patience': [20, 30, 40],
            'hidden_layers': [
                [(4,)],
                [(6,)],
                [(8,)],
                [(16,)],
                [(4,), (4,)],
                [(8,), (8,)],
                [(16,), (8,)],
                [(32,), (16,)],
            ],
            'batch_size': [1, 8, 16, 32],
            'momentum': [0.0, 0.5, 0.9],
            'mu': [1.5, 1.75, 2.0]
        }
        
        # Filtra i parametri non necessari per certi algoritmi
        all_combinations = []
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            
            # Filtra combinazioni inappropriate
            if params['algorithm'] == 'sgd' and params['momentum'] == 0.0:
                # Per SGD senza momentum, non serve mu
                params['mu'] = 1.75  # Valore di default
                all_combinations.append(params)
            elif params['algorithm'] in ['rprop', 'quickprop']:
                # Per RPROP e Quickprop, non usiamo momentum
                params['momentum'] = 0.0
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
        """Grid search di raffinamento intorno ai migliori parametri"""
        print("\n" + "="*80)
        print("FINE GRID SEARCH (Raffinamento)")
        print("="*80)
        
        # Definisci intervalli di raffinamento per ciascun parametro
        fine_tuning_grid = {
            'eta': [best_params['eta'] * 0.5, best_params['eta'], best_params['eta'] * 1.5],
            'l2_lambda': [best_params['l2_lambda'] * 0.5, best_params['l2_lambda'], best_params['l2_lambda'] * 1.5],
            'decay': [max(0.5, best_params['decay'] - 0.05), best_params['decay'], min(0.99, best_params['decay'] + 0.05)],
            'batch_size': [max(1, best_params['batch_size'] - 4), best_params['batch_size'], best_params['batch_size'] + 4],
        }
        
        # Aggiungi parametri specifici dell'algoritmo
        if best_params['algorithm'] == 'rprop':
            fine_tuning_grid['eta_plus'] = [best_params['eta_plus'] - 0.1, best_params['eta_plus'], best_params['eta_plus'] + 0.1]
            fine_tuning_grid['eta_minus'] = [best_params['eta_minus'] - 0.1, best_params['eta_minus'], best_params['eta_minus'] + 0.1]
        elif best_params['algorithm'] == 'quickprop':
            fine_tuning_grid['mu'] = [best_params['mu'] - 0.25, best_params['mu'], best_params['mu'] + 0.25]
        
        # Testa anche diverse strutture dai migliori risultati
        hidden_layers_options = []
        for params, _ in top_results:
            if params['hidden_layers'] not in hidden_layers_options:
                hidden_layers_options.append(params['hidden_layers'])
        
        fine_tuning_grid['hidden_layers'] = hidden_layers_options
        
        # Mantieni gli altri parametri fissi
        fixed_params = {
            'epochs': best_params['epochs'],
            'algorithm': best_params['algorithm'],
            'activation_type': best_params['activation_type'],
            'weight_initializer': best_params['weight_initializer'],
            'loss_type': best_params['loss_type'],
            'patience': best_params['patience'],
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
        
        # Grafico 1: Training e Validation Loss
        axes[0].plot(history['training'], label='Training Loss', alpha=0.7)
        axes[0].plot(history['validation'], label='Validation Loss', alpha=0.7)
        axes[0].set_xlabel('Epoche')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training e Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Trova il punto di minima validation loss
        min_val_loss_epoch = np.argmin(history['validation'])
        min_val_loss = history['validation'][min_val_loss_epoch]
        axes[0].axvline(x=min_val_loss_epoch, color='r', linestyle='--', alpha=0.5, 
                       label=f'Min Val Loss: {min_val_loss:.4f}')
        axes[0].legend()
        
        # Grafico 2: Bar plot delle accuracy
        if self.dataset_name in ['monk1', 'monk2', 'monk3']:
            metrics = ['Train Accuracy', 'Val Accuracy', 'Test Accuracy']
            values = [results['train_accuracy'], results['val_accuracy'], results['test_accuracy']]
            
            bars = axes[1].bar(metrics, values, color=['blue', 'green', 'red'])
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy sui diversi set')
            axes[1].set_ylim([0, 1])
            
            # Aggiungi i valori sulle barre
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        else:
            # Per regressione, mostra le loss
            metrics = ['Train Loss', 'Val Loss', 'Test Loss']
            values = [results['train_loss'], results['val_loss'], results['test_loss']]
            
            bars = axes[1].bar(metrics, values, color=['blue', 'green', 'red'])
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss sui diversi set')
            
            # Aggiungi i valori sulle barre
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Salva il grafico
        filename = f'grid_search_results_{self.dataset_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
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
        dataset_name = 'monk1'  # Cambia questo per testare dataset diversi
    
    # Crea e esegui la grid search
    # datasets = ['monk1', 'monk2', 'monk3', 'cup']
    # for dataset in datasets:
    #    print(f"\n\n{'#'*80}")
    #    print(f"ESECUZIONE PER DATASET: {dataset}")
    #    print(f"{'#'*80}")
    #    grid_search = GridSearch(dataset_name=dataset)
        grid_search = GridSearch(dataset_name=dataset_name)
        grid_search.run()


if __name__ == "__main__":
    main()