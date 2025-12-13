import numpy as np
import neuron as nrn

class CascadeNetwork:
    def __init__(self, n_inputs, n_outputs, learning_rate=0.1, algorithm="quickprop"):
        """
        Inizializza la rete per la Fase 0 in cui non c'è nessuna unità nascosta.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        
        # La classe CascadeNetwork incapsula i neuroni di output 
        # e il costruttore provvede a costruire i neuroni autonomamente
        # secondo le regole della Cascade Correlation
        self.output_neurons = []
        for i in range(n_outputs): # Nota che il peso del bias è già incluso nei pesi del neurone perché 
                                   # viene creato e gestito nel costruttore della classe Neuron
                                   # Matematicamente: (Input * Pesi) + self.bias  <-- Equivale a --> (Input * Pesi) + (1 * PesoBias)
            neuron = nrn.Neuron(num_inputs=n_inputs, # All'inizio non ci sono neuroni nascosti (il bias è gestito a parte dalla classe Neuron)
                                index_in_layer=i, # Per tenere traccia della posizione del neurone nelle connessioni future
                                is_output_neuron=True, # Verifica che sia un neurone di output per il calcolo del delta
                                activation_function_type="tanh") # Come funzione di attivazione si usa la tangente iperbolica (tanh)
                                                                 # che mappa gli input in un range tra -1 e 1 e include già il 
                                                                 # calcolo della derivata della funzione di attivazione 
            self.output_neurons.append(neuron)
        
        # Qui verranno salvati i neuroni nascosti più avanti
        # e sarà l'algoritmo in modo autonomo a decidere se e quanti aggiungerne
        self.hidden_units = []

    def forward(self, input_pattern): # input_pattern è un array di dati di input per un singolo esempio
        outputs = []
        for n in self.output_neurons:
            out = n.feed_neuron(input_pattern) # Delega il calcolo della predizione al neurone stesso
            outputs.append(out)
        return np.array(outputs)
    
    def train_phase_0(self, X, y, epochs=1000, tolerance=0.01, patience=20):
        """
        Addestramento dei pesi Input -> Output.
        X: Dataset di input
        y: Target 
        """
        min_error = float('inf') # Inizializza il minimo errore a infinito
        patience_counter = 0 # Contatore per la logica di early stopping
        batch_size = len(X) # Usiamo full-batch perché la Cascade Correlation lo richiede   

        for epoch in range(epochs):
            total_error = 0.0 # L'errore viene resettato all'inizio di ogni epoca 
            
            for neuron in self.output_neurons: # All'inizio di ogni epoca i gradienti di ogni neurone vengono resettati
                neuron.reset_grad_accum()
            
            for i in range(batch_size):
                input_pattern = X[i]
                target_pattern = y[i] 
                
                outputs = self.forward(input_pattern)
    
                for idx, neuron in enumerate(self.output_neurons): # enumerate per ottenere l'indice del neurone di output
                    if isinstance(target_pattern, (list, np.ndarray)): # isinstance controlla che target_pattern sia una lista o un array numpy
                        target = target_pattern[idx] # Estrae il target corrispondente all'output del neurone idx
                    else:
                        target = target_pattern # Se c'è un solo output, usa direttamente il target
                    output = outputs[idx] # Estrae l'output del neurone numero idx
                    
                    error_signal = target - output # Calcolo dell'errore
                    total_error += 0.5 * (error_signal ** 2) # Accumula l'errore quadratico medio
                    neuron.compute_delta(error_signal) # Calcola il delta per il neurone di output
                    neuron.compute_gradients(input_pattern) # Calcola i gradienti per il neurone di output
                    neuron.accumulate_gradients() # Accumula i gradienti per il neurone di output

            for neuron in self.output_neurons: # Dopo aver processato tutto il batch, aggiorna i pesi
                neuron.apply_accumulated_gradients(
                    eta=self.learning_rate,
                    batch_size=batch_size,
                    algorithm=self.algorithm
                )
            
            if epoch % 100 == 0: # Stampa l'errore ogni 100 epoche
                print(f"Epoch {epoch}: Error = {total_error:.5f}")

            if total_error < min_error - tolerance: # Il miglioramento deve essere maggiore della tolleranza
                min_error = total_error
                patience_counter = 0 # Reset del conteggio dei fallimenti
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Convergenza raggiunta (o stallo) all'epoca {epoch}. Errore finale: {total_error:.5f}")
                break
        
        return total_error # Per capire se la Cascade Correlation deve aggiungere un nuovo neurone nascosto (lo aggiunge quando l'errore è alto)
    
    def _compute_residuals(self, X, y):
        """
        Calcola l'errore residuo per ogni pattern nel dataset.
        """
        outputs = []
        for i in range(len(X)):
            out = self.forward(X[i])
            outputs.append(out)
        outputs = np.array(outputs)
        residuals = y - outputs
        return residuals
    
    def _install_candidate(self, candidate_neuron):
        """
        Aggiunge il neurone candidato alla rete come nuovo neurone nascosto.
        """
        self.hidden_units.append(candidate_neuron)
        for out_neuron in self.output_neurons:
            out_neuron.weights = np.append(out_neuron.weights, 0.0)
            
            if hasattr(out_neuron, 'prev_weight_updates'):
                out_neuron.prev_weight_updates = np.append(out_neuron.prev_weight_updates, 0.0)
            if hasattr(out_neuron, 'prev_gradients'):
                out_neuron.prev_gradients = np.append(out_neuron.prev_gradients, 0.0)
                
        print(f"Neurone nascosto aggiunto. Totale neuroni nascosti: {len(self.hidden_units)}")
        
    def train(self, X, y, max_epochs=2000, tolerance=0.01, patience=50, max_hidden_units=10):
        """
        Metodo di addestramento principale che include la logica per aggiungere neuroni nascosti.
        """
        print("Inizio addestramento (fase 0) della rete Cascade Correlation...")
        final_error = self.train_phase_0(X, y, epochs=max_epochs, tolerance=tolerance, patience=patience) # Addestramento della fase 0

        while current_error > tolerance and len(self.hidden_units) < max_hidden_units: # Logica per aggiungere neuroni nascosti
            print(f"Aggiunta neurone nascosto {len(self.hidden_units) + 1}")
            print(f"Errore attuale: {current_error:.5f}. Obiettivo: {tolerance}")
            
            residuals = self._compute_residuals(X, y) # Calcola l'errore residuo
            best_candidate = self._train_candidates(X, residuals) # Cerchiamo il neurone che meglio si adatta all'errore residuo
            self._install_candidate(best_candidate) # Aggiungiamo il neurone alla rete
            
            print("Riadattamento dei pesi output dopo l'aggiunta del neurone nascosto...")
            current_error = self.train_phase_0(X, y, epochs=max_epochs, tolerance=tolerance, patience=patience) 
            
            if current_error < tolerance:
                print("SUCCESSO: Obiettivo di errore raggiunto dopo l'aggiunta dei neuroni nascosti.")
            else:
                print(f"FALLIMENTO: Obiettivo di errore non raggiunto e raggiunto numero massimo di neuroni ({max_hidden_units}).")
            return current_error
        return final_error
    
    
    
    def _train_candidates(self, X, residuals, n_candidates=8, epochs=100):
        """
        Allena più neuroni candidati e restituisce quello con la migliore correlazione con l'errore residuo.
        """
        from neuron import Neuron
        if len(self.hidden_units) > 0:
            hidden_outputs = []
            for h_unit in self.hidden_units:
                h_out = np.array([h_unit.forward(p) for p in X])
                hidden_outputs.append(h_out)
            hidden_outputs = np.column_stack(hidden_outputs)
            augmented_X = np.hstack([X, hidden_outputs])
        else:
            augmented_X = X
        
        input_dim = augmented_X.shape[1]
        candidates = [Neuron(input_dim, activation="tanh") for _ in range (n_candidates)]
        cand_lr = self.learning_rate
        avg_residuals = np.mean(residuals, axis =0)
        
        for epoch in range(epochs):
            for candidate in candidates:
                values = np.array([candidate.forward(x) for x in augmented_X])
                avg_value = np.mean(values)
                candidate_diff = values - avg_value
                residual_diff = residuals - avg_residuals
                correlations = np.sum(candidate_diff[:, None] * residual_diff, axis=0)
                signs = np.sign(correlations)
                correlation_term = np.sum(signs*residual_diff, axis=1)
                act_derivative = 1.0 - values**2
                delta = correlation_term*act_derivative
                weight_changes = np.dot(augmented_X.T, delta)
                candidate.weights += cand_lr*weight_changes
                candidate.bias += cand_lr*np.sum(delta)
        
        best_candidate = None
        best_score = -1.0
        
        for candidate in candidates:
            values = np.array([candidate.forward(x) for x in augmented_X])
            score = self._compute_correlation_score(values, residuals)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        print(f"Miglior candidato selezionato con punteggio di correlazione: {best_score:.5f}")
        return best_candidate
    
    def _compute_correlation_score(self, candidate_values, residuals):
        """
        Calcola il punteggio di correlazione tra le uscite del candidato e l'errore residuo.
        """
        v_mean = np.mean(candidate_values)
        r_mean = np.mean(residuals, axis=0)
        
        term1= candidate_values - v_mean
        term2 = residuals - r_mean
        
        covariances = np.sum(term1[:, None] * term2, axis=0)
        
        S = np.sum(np.abs(covariances))
        return S