import numpy as np
import neuron as nrn

class CascadeNetwork:
    def __init__(self, n_inputs, n_outputs, learning_rate=0.1, algorithm="quickprop", l2_lambda=0.0):
        """
        Inizializza la rete per la Fase 0 in cui non c'è nessuna unità nascosta.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.l2_lambda = l2_lambda
        
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
        self.loss_history = []
        self.val_loss_history = []

    def forward(self, input_pattern):
        """
        Calcola l'output. 
        """
        current_input = np.array(input_pattern)
    
        for h_unit in self.hidden_units:
            n_weights = len(h_unit.weights)
            relevant_input = current_input[:n_weights]
            
            h_out = h_unit.feed_neuron(relevant_input)
            
            current_input = np.append(current_input, h_out)

        outputs = []
        for n in self.output_neurons:
            out = n.feed_neuron(current_input)
            outputs.append(out)
            
        return np.array(outputs)
    
    def calculate_loss(self, X, y):
        """Calcola la loss (MSE) su un dato set (es. Validation)"""
        if len(X) == 0: return 0.0
        
        total_error = 0.0
        for i in range(len(X)):
            pattern = X[i]
            target_pattern = y[i]
            outputs = self.forward(pattern)
            
            for idx in range(len(self.output_neurons)):
                if isinstance(target_pattern, (list, np.ndarray)):
                    target = target_pattern[idx]
                else:
                    target = target_pattern
                output = outputs[idx]
                total_error += 0.5 * (target - output) ** 2
        
        return total_error / len(X)
    
    def train_phase_0(self, X, y, X_val=None, y_val=None, epochs=1000, tolerance=0.01, patience=20):
        """
        Addestramento dei pesi Input -> Output.
        X: Dataset di input
        y: Target 
        """
        
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        if not hasattr(self, 'val_loss_history'):
            self.val_loss_history = []
            
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
                    output = outputs[idx]
                    error_signal = target - output
                    neuron.accumulate_gradients() # Accumula i gradienti per il neurone di output
            
            current_train_loss = total_error / batch_size
            self.loss_history.append(current_train_loss) # Salva l'errore medio per epoca
            
            if X_val is not None and y_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                self.val_loss_history.append(val_loss)

            for neuron in self.output_neurons: # Dopo aver processato tutto il batch, aggiorna i pesi
                neuron.apply_accumulated_gradients(
                    eta=self.learning_rate,
                    batch_size=batch_size,
                    algorithm=self.algorithm,
                    l2_lambda=self.l2_lambda
                )
            
            if epoch % 100 == 0: # Stampa l'errore ogni 100 epoche
                print(f"Epoch {epoch}: Error = {total_error:.5f}")

            if total_error < min_error - tolerance: # Il miglioramento deve essere maggiore della tolleranza
                min_error = total_error
                patience_counter = 0 # Reset del conteggio dei fallimenti
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                # print(f"Convergenza raggiunta (o stallo) all'epoca {epoch}. Errore finale: {total_error:.5f}")
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
            out_neuron.weight_grad_accum = np.append(out_neuron.weight_grad_accum, 0.0)
            if hasattr(out_neuron, 'vel_w'):
                out_neuron.vel_w = np.append(out_neuron.vel_w, 0.0)
            if hasattr(out_neuron, 'prev_weight_grad'):
                out_neuron.prev_weight_grad = np.append(out_neuron.prev_weight_grad, 0.0)
            if hasattr(out_neuron, 'prev_weight_update'):
                out_neuron.prev_weight_update = np.append(out_neuron.prev_weight_update, 0.0)
            if hasattr(out_neuron, 'rprop_step_w'):
                out_neuron.rprop_step_w = np.append(out_neuron.rprop_step_w, 0.1) 
                
        print(f"Neurone nascosto aggiunto. Totale neuroni nascosti: {len(self.hidden_units)}")
        
    def train(self, X, y, X_val=None, y_val=None, max_epochs=2000, tolerance=0.01, patience=50, max_hidden_units=10):
        
        self.loss_history = []
        self.val_loss_history = []
        
        print("Inizio addestramento (Fase 0)...")
        current_error = self.train_phase_0(X, y, X_val=X_val, y_val=y_val, epochs=max_epochs, tolerance=tolerance, patience=patience)
        print(f"Fase 0 completata. Errore: {current_error:.5f}")
        
        while current_error > tolerance and len(self.hidden_units) < max_hidden_units:
            print(f"\n>>> Aggiunta neurone nascosto #{len(self.hidden_units) + 1} (Err: {current_error:.5f}) <<<")
            residuals = self._compute_residuals(X, y)
            best_candidate = self._train_candidates(X, residuals)
            self._install_candidate(best_candidate)
            print("Riadattamento pesi output...")
            current_error = self.train_phase_0(X, y, X_val=X_val, y_val=y_val, epochs=max_epochs, tolerance=tolerance, patience=patience//2) 
            
            if current_error < tolerance:
                print("SUCCESSO: Obiettivo raggiunto!")
                break
        
        return current_error
    
    
    
    def _train_candidates(self, X, residuals, n_candidates=8, epochs=500):
        """
        Allena i candidati usando Gradient Ascent con MOMENTUM. Normalizza i gradienti per la dimensione del batch.
        """
        if len(self.hidden_units) > 0:
            hidden_outputs_matrix = []
            for input_p in X:
                current_pattern_extended = np.array(input_p) 
                
                for h in self.hidden_units:
                    n_w = len(h.weights)
                    out_val = h.feed_neuron(current_pattern_extended[:n_w])
                    current_pattern_extended = np.append(current_pattern_extended, out_val)
                hidden_outputs_matrix.append(current_pattern_extended[len(input_p):]) 
            
            hidden_outputs_matrix = np.array(hidden_outputs_matrix)
            augmented_X = np.hstack([X, hidden_outputs_matrix])
        else:
            augmented_X = X
            
        input_dim = augmented_X.shape[1]
        batch_size = augmented_X.shape[0] 
        
        candidates = []
        
        velocities_w = [] 
        velocities_b = []
        
        for i in range(n_candidates):
            cand = nrn.Neuron(num_inputs=input_dim, index_in_layer=0, is_output_neuron=False, activation_function_type="tanh")
            cand.weights = np.random.uniform(-0.5, 0.5, size=input_dim)
            cand.bias = np.random.uniform(-0.5, 0.5)
            candidates.append(cand)
            velocities_w.append(np.zeros(input_dim))
            velocities_b.append(0.0)
            
        cand_lr = self.learning_rate 
        momentum = 0.9 
        
        avg_residuals = np.mean(residuals, axis=0)

        for epoch in range(epochs):
            for i, candidate in enumerate(candidates):
                values = np.array([candidate.feed_neuron(x) for x in augmented_X])
    
                avg_value = np.mean(values)
                candidate_diff = values - avg_value
                residual_diff = residuals - avg_residuals
             
                covariances = np.dot(candidate_diff, residual_diff) 
                signs = np.sign(covariances)
                
                correlation_error = np.sum(signs * residual_diff, axis=1) 
                act_deriv = 1.0 - values**2 
                delta = correlation_error * act_deriv
                
                grad_w = np.dot(augmented_X.T, delta)
                grad_b = np.sum(delta)
                
                grad_w /= batch_size
                grad_b /= batch_size
                
                velocities_w[i] = (momentum * velocities_w[i]) + (cand_lr * grad_w)
                velocities_b[i] = (momentum * velocities_b[i]) + (cand_lr * grad_b)
                
                candidate.weights += velocities_w[i]
                candidate.bias += velocities_b[i]
        
        best_candidate = None
        best_score = -1.0
        
        for candidate in candidates:
            values = np.array([candidate.feed_neuron(x) for x in augmented_X])
            score = self._compute_correlation_score(values, residuals)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                
        print(f"Miglior candidato selezionato. Score: {best_score:.5f}")
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