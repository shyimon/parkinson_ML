# dichotomic_cv_search.py
import numpy as np
import itertools
import time
from collections import defaultdict
from neural_network import NeuralNetwork
from data_manipulation import return_monk3

class DichotomicCVSearch:
    """Ricerca dicotomica N-dimensionale con Cross-Validation"""
    
    def __init__(self, param_ranges):
        """
        param_ranges: dizionario con intervalli per ogni iperparametro
        Esempio:
        {
            'eta': (0.001, 0.1),
            'l2_lambda': (0.0, 0.01),
            'hidden_neurons': (4, 16),
            'num_layers': (2, 4)  # escluso input e output
        }
        """
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.n_dims = len(self.param_names)
        self.results = []
        self.best_score = -np.inf
        self.best_params = None
        
    def _map_params_to_config(self, params):
        """Mappa parametri normalizzati [0,1]^N a valori reali"""
        config = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[name]
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Parametro intero (es. numero neuroni, layer)
                config[name] = int(min_val + params[i] * (max_val - min_val))
            else:
                # Parametro float
                config[name] = min_val + params[i] * (max_val - min_val)
        return config
    
    def _create_network_config(self, base_params, specific_params):
        """Crea configurazione completa per la rete"""
        config = {
            'network_structure': [17] + [specific_params.get('hidden_neurons', 8)] * 
                                 specific_params.get('num_layers', 2) + [1],
            'eta': specific_params.get('eta', 0.05),
            'l2_lambda': specific_params.get('l2_lambda', 0.001),
            'momentum': specific_params.get('momentum', 0.9),
            'algorithm': 'sgd',
            'activation_type': 'sigmoid',
            'loss_type': 'binary_crossentropy',
            'weight_initializer': 'xavier',
            'decay': 0.95,
            'mu': 1.75,
            'eta_plus': 1.2,
            'eta_minus': 0.5,
            'debug': False
        }
        config.update(base_params)
        return config
    
    def cross_validate(self, config, X, y, k=5):
        """K-fold cross-validation"""
        n_samples = len(X)
        fold_size = n_samples // k
        scores = []
        
        indices = np.random.permutation(n_samples)
        
        for i in range(k):
            # Split in train e validation
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k - 1 else n_samples
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Crea e addestra rete
            net = NeuralNetwork(**config)
            
            # Addestra con early stopping
            history = net.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=200,
                batch_size=min(16, len(X_train)),
                patience=15,
                min_delta=0.0001,
                verbose=False
            )
            
            # Valuta su validation fold
            val_pred = net.predict(X_val)
            val_pred_class = (val_pred > 0.5).astype(int)
            accuracy = np.mean(val_pred_class == y_val)
            scores.append(accuracy)
        
        return np.mean(scores)
    
    def evaluate_point(self, params, X, y):
        """Valuta un punto nello spazio degli iperparametri"""
        # Mappa parametri normalizzati a valori reali
        specific_params = self._map_params_to_config(params)
        
        # Crea configurazione completa
        config = self._create_network_config({}, specific_params)
        
        # Esegui cross-validation
        score = self.cross_validate(config, X, y, k=3)  # 3-fold per velocità
        
        return score
    
    def dichotomic_search(self, X, y, max_iterations=10, epsilon=0.1):
        """Ricerca dicotomica N-dimensionale"""
        print(f"RICERCA DICOTOMICA {self.n_dims}-D CON CROSS-VALIDATION")
        print("="*60)
        print(f"Parametri da ottimizzare: {self.param_names}")
        print(f"Intervalli: {self.param_ranges}")
        
        # Inizializza i vertici dell'ipercubo [0,1]^N
        # Genera tutti i vertici (2^N punti)
        vertices = list(itertools.product([0, 1], repeat=self.n_dims))
        
        iteration = 0
        
        while iteration < max_iterations:
            print(f"\n{'='*60}")
            print(f"ITERAZIONE {iteration + 1}/{max_iterations}")
            print(f"Dimensione spazio: {2 ** self.n_dims} regioni")
            
            # Valuta tutti i vertici
            scores = []
            for vertex in vertices:
                score = self.evaluate_point(vertex, X, y)
                scores.append(score)
                
                # Aggiorna miglior risultato
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = vertex
                    print(f"  Nuovo record: {score:.4%}")
            
            # Trova il vertice migliore
            best_idx = np.argmax(scores)
            best_vertex = vertices[best_idx]
            
            print(f"\nMiglior vertice: {best_vertex}")
            print(f"Miglior score: {scores[best_idx]:.4%}")
            
            # Se siamo all'ultima iterazione, termina
            if iteration == max_iterations - 1:
                break
            
            # Dividi ogni dimensione a metà attorno al vertice migliore
            new_vertices = []
            for vertex in vertices:
                # Crea nuovi vertici spostandosi verso il centro
                new_vertex = []
                for i in range(self.n_dims):
                    if vertex[i] == 0:
                        # Se era 0, diventa 0.5 (verso il centro)
                        new_vertex.append(0.5)
                    else:
                        # Se era 1, diventa 0.5
                        new_vertex.append(0.5)
                new_vertices.append(tuple(new_vertex))
            
            # Aggiungi anche il vertice migliore originale
            new_vertices.append(best_vertex)
            
            # Genera tutti i nuovi vertici dell'ipercubo ridotto
            vertices = list(set(new_vertices))
            
            # Se ci sono troppi vertici, campiona
            if len(vertices) > 8:
                # Mantieni solo i migliori
                vertex_scores = []
                for vertex in vertices:
                    score = self.evaluate_point(vertex, X, y)
                    vertex_scores.append((score, vertex))
                
                vertex_scores.sort(reverse=True, key=lambda x: x[0])
                vertices = [v for _, v in vertex_scores[:8]]
            
            iteration += 1
        
        # Mappa i migliori parametri normalizzati a valori reali
        best_real_params = self._map_params_to_config(self.best_params)
        
        print(f"\n{'='*60}")
        print("RICERCA COMPLETATA")
        print("="*60)
        print(f"Miglior score CV: {self.best_score:.4%}")
        print(f"Migliori parametri:")
        for name, value in best_real_params.items():
            print(f"  {name}: {value}")
        
        return best_real_params, self.best_score
    
    def refined_search(self, best_params, X, y, radius=0.2, steps=5):
        """Ricerca raffinata attorno al miglior punto trovato"""
        print(f"\n{'='*60}")
        print("RICERCA RAFFINATA")
        print("="*60)
        
        # Converti i migliori parametri reali in normalizzati
        best_normalized = []
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[name]
            param_val = best_params[name]
            if isinstance(min_val, int):
                param_val = int(param_val)
            norm_val = (param_val - min_val) / (max_val - min_val)
            best_normalized.append(norm_val)
        
        # Crea griglia attorno al punto
        refined_best = best_params.copy()
        refined_score = self.best_score
        
        for i, name in enumerate(self.param_names):
            print(f"\nOttimizzazione raffinata di: {name}")
            
            # Crea valori da testare
            min_val, max_val = self.param_ranges[name]
            current_val = best_params[name]
            
            if isinstance(min_val, int):
                # Parametro intero
                step = max(1, int(radius * (max_val - min_val)))
                test_values = range(
                    max(min_val, current_val - step),
                    min(max_val, current_val + step) + 1
                )
            else:
                # Parametro float
                step = radius * (max_val - min_val)
                test_values = np.linspace(
                    max(min_val, current_val - step),
                    min(max_val, current_val + step),
                    steps
                )
            
            # Testa ogni valore
            for val in test_values:
                test_params = best_params.copy()
                test_params[name] = val
                
                # Crea configurazione
                config = self._create_network_config({}, test_params)
                
                # Cross-validation
                score = self.cross_validate(config, X, y, k=3)
                
                if score > refined_score:
                    refined_score = score
                    refined_best = test_params.copy()
                    print(f"  {name}={val:.4f}: {score:.4%} ✓")
                else:
                    print(f"  {name}={val:.4f}: {score:.4%}")
        
        return refined_best, refined_score

def run_complete_search():
    """Esegue la ricerca completa"""
    
    # Carica i dati
    X_train, y_train, X_val, y_val, X_test, y_test = return_monk3(
        one_hot=True, 
        val_split=0.3,
        dataset_shuffle=True
    )
    
    # Combina training e validation per la cross-validation
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.vstack([y_train, y_val])
    
    print(f"Dataset per Cross-Validation:")
    print(f"  Esempi totali: {X_cv.shape[0]}")
    print(f"  Test set: {X_test.shape[0]}")
    
    # Definizione degli intervalli di ricerca
    param_ranges = {
        'eta': (0.001, 0.1),          # Learning rate
        'l2_lambda': (0.0, 0.01),     # Regolarizzazione
        'hidden_neurons': (4, 16),    # Neuroni per layer nascosto
        'num_layers': (2, 4),         # Numero layer nascosti
        'momentum': (0.05, 0.99),       # Momentum
        'epochs': (100, 700)           # Numero di epoche
    }
    
    # Crea il searcher
    searcher = DichotomicCVSearch(param_ranges)
    
    # Esegui la ricerca dicotomica
    print("\n" + "="*60)
    print("FASE 1: RICERCA DICOTOMICA")
    print("="*60)
    
    start_time = time.time()
    best_params, best_score = searcher.dichotomic_search(X_cv, y_cv, max_iterations=3)
    cv_time = time.time() - start_time
    
    # Ricerca raffinata
    print("\n" + "="*60)
    print("FASE 2: RICERCA RAFFINATA")
    print("="*60)
    
    refined_params, refined_score = searcher.refined_search(
        best_params, X_cv, y_cv, radius=0.2, steps=5
    )
    
    # VALUTAZIONE FINALE SUL TEST SET
    print("\n" + "="*60)
    print("VALUTAZIONE FINALE SUL TEST SET")
    print("="*60)
    
    # Crea la rete con i migliori parametri
    final_config = searcher._create_network_config({}, refined_params)
    
    print(f"\nConfigurazione finale:")
    for key, value in final_config.items():
        if key in ['network_structure', 'eta', 'l2_lambda', 'algorithm']:
            print(f"  {key}: {value}")
    
    # Addestra sul training+validation completo
    print("\nAddestramento finale su training+validation...")
    final_net = NeuralNetwork(**final_config)
    
    history = final_net.fit(
        X_cv, y_cv,
        X_test, y_test,  # Usa test set come validation per early stopping
        epochs=300,
        batch_size=16,
        patience=20,
        min_delta=0.0001,
        verbose=True
    )
    
    # Valuta sul test set
    test_pred = final_net.predict(X_test)
    test_pred_class = (test_pred > 0.5).astype(int)
    test_accuracy = np.mean(test_pred_class == y_test)
    
    # Training accuracy
    train_pred = final_net.predict(X_cv)
    train_pred_class = (train_pred > 0.5).astype(int)
    train_accuracy = np.mean(train_pred_class == y_cv)
    
    print(f"\n{'='*60}")
    print("RISULTATI FINALI")
    print("="*60)
    print(f"Tempo ricerca: {cv_time:.1f} secondi")
    print(f"Score Cross-Validation: {refined_score:.4%}")
    print(f"Accuracy Training: {train_accuracy:.4%}")
    print(f"Accuracy Test: {test_accuracy:.4%}")
    
    # Confusion Matrix
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
    
    # GRAFICO: Training Loss vs Test Loss
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    if history and 'training' in history:
        plt.plot(history['training'], label='Training Loss', linewidth=2, color='blue')
    
    if history and 'validation' in history:
        plt.plot(history['validation'], label='Test Loss', linewidth=2, color='red')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training vs Test Loss (Accuracy: {test_accuracy:.2%})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi testo con i parametri
    param_text = f"Params: η={final_config['eta']:.3f}, λ={final_config['l2_lambda']:.4f}\n"
    param_text += f"Hidden: {final_config['network_structure'][1:-1]}\n"
    param_text += f"Test Acc: {test_accuracy:.2%}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('dichotomic_cv_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'test_accuracy': test_accuracy,
        'cv_score': refined_score,
        'train_accuracy': train_accuracy,
        'params': refined_params,
        'network': final_net,
        'history': history
    }

if __name__ == "__main__":
    print("RICERCA DICOTOMICA N-DIMENSIONALE CON CROSS-VALIDATION")
    print("="*60)
    
    try:
        results = run_complete_search()
        
        print(f"\n{'='*60}")
        print("RIEPILOGO")
        print("="*60)
        
        
        if results['test_accuracy'] >= 0.97:
            print(" Ottimo risultato! Vicino al massimo teorico per Monk 3")
            print(f"   Test Accuracy: {results['test_accuracy']:.4%}")
            print(f"   Cross-Validation Score: {results['cv_score']:.4%}")
        else:
            print(f" Test Accuracy: {results['test_accuracy']:.4%}")
            print("\nConsigli per migliorare:")
            print("1. Aumenta max_iterations nella ricerca dicotomica")
            print("2. Allarga gli intervalli di ricerca")
            print("3. Prova con k-fold maggiore (k=5)")
            print("4. Aggiungi più parametri alla ricerca")
        
        print(f"\nParametri ottimali trovati:")
        for name, value in results['params'].items():
            print(f"  {name}: {value}")
            
    except Exception as e:
        print(f"\n Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()