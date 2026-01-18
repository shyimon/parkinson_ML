import numpy as np
import neuron as nrn

class CascadeNetwork:
    def __init__(self, n_inputs, n_outputs, learning_rate=0.01, algorithm="quickprop", l2_lambda=0.0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.l2_lambda = l2_lambda
        
        self.output_neurons = []
        for i in range(n_outputs): 
            # Output neuron initialization
            neuron = nrn.Neuron(num_inputs=n_inputs, 
                                index_in_layer=i, 
                                is_output_neuron=True, 
                                activation_function_type="tanh") # Output range [-1, 1] per CUP normalizzato
            self.output_neurons.append(neuron)
        
        self.hidden_units = []
        self.loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []

    def forward(self, input_pattern):
        current_input = np.array(input_pattern, dtype=float)
    
        # Passaggio attraverso i neuroni nascosti esistenti
        for h_unit in self.hidden_units:
            n_weights = len(h_unit.weights)
            relevant_input = current_input[:n_weights]
            
            h_out = h_unit.feed_neuron(relevant_input)
        
            current_input = np.append(current_input, h_out)

        # Calcolo output finale
        outputs = []
        for n in self.output_neurons:
            out = n.feed_neuron(current_input)
            outputs.append(out)
            
        return np.array(outputs)
    
    def _install_candidate(self, candidate_neuron):
        """Aggiunge il neurone candidato alla rete."""
        candidate_neuron.index_in_layer = len(self.hidden_units) 
        self.hidden_units.append(candidate_neuron)
      
        for out_neuron in self.output_neurons:
            new_weight = np.random.uniform(-0.1, 0.1)
            out_neuron.weights = np.append(out_neuron.weights, new_weight)
            
            out_neuron.weight_grad_accum = np.zeros_like(out_neuron.weights)
            
            candidate_neuron.attach_to_output(self.output_neurons)
                
        print(f"Neurone nascosto installato. Totale: {len(self.hidden_units)}")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, max_epochs=1000, tolerance=1e-5, patience=20, max_hidden_units=10):
        self.loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        
        inputs_train = np.array(X_train)
        inputs_val = np.array(X_val) if X_val is not None else None
        inputs_test = np.array(X_test) if X_test is not None else None
  
        print("Fase 0: Addestramento iniziale (Input -> Output)...")
        
        self._train_output_weights(inputs_train, y_train, inputs_val, y_val, inputs_test, y_test, max_epochs, tolerance, patience)
        
        while len(self.hidden_units) < max_hidden_units:
            # Calcolo residui
            predictions = np.array([self.forward(x) for x in X_train])
            residuals = y_train - predictions
            mse = np.mean(residuals**2)
            
            print(f"--- Hidden Unit {len(self.hidden_units)+1} Search. Current MSE: {mse:.5f} ---")
            
            if mse <= tolerance:
                print("Tolleranza raggiunta.")
                break
             
            best_candidate = self._train_candidates(inputs_train, residuals, n_candidates=8, epochs=max_epochs//2)
            
            if best_candidate is None:
                print("Fallimento nel trovare un candidato utile.")
                break
            
            # Calcoliamo le uscite del nuovo neurone per aggiungerle al dataset
            new_feat_train = np.array([best_candidate.feed_neuron(x) for x in inputs_train]).reshape(-1, 1)
            inputs_train = np.hstack((inputs_train, new_feat_train))
            
            if inputs_val is not None:
                new_feat_val = np.array([best_candidate.feed_neuron(x) for x in inputs_val]).reshape(-1, 1)
                inputs_val = np.hstack((inputs_val, new_feat_val))
                
            if inputs_test is not None:
                new_feat_test = np.array([best_candidate.feed_neuron(x) for x in inputs_test]).reshape(-1, 1)
                inputs_test = np.hstack((inputs_test, new_feat_test))

            self._install_candidate(best_candidate)
            
            print(f"Riadattamento Output Layer...")
            self._train_output_weights(inputs_train, y_train, inputs_val, y_val, inputs_test, y_test, max_epochs, tolerance, patience)
            
        return self.loss_history[-1] if self.loss_history else 0.0
    
    def _train_output_weights(self, X, y, X_val, y_val, X_test, y_test, epochs, tolerance, patience):
        """Allena SOLO i pesi dei neuroni di output."""
        best_loss = float('inf')
        patience_cnt = 0
        
        # Salvataggio pesi
        best_w = [n.weights.copy() for n in self.output_neurons]
        best_b = [n.bias for n in self.output_neurons]
        
        for epoch in range(epochs):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
            
                outputs = []
                for n in self.output_neurons:
                    outputs.append(n.feed_neuron(inputs))
                outputs = np.array(outputs)
                
                errors = target - outputs
                
                for j, n in enumerate(self.output_neurons):
                    n.compute_delta(errors[j]) 
            
            for n in self.output_neurons:
                n.update_weights(self.learning_rate, self.l2_lambda)
            
            curr_preds = np.array([[n.feed_neuron(X[k]) for n in self.output_neurons] for k in range(len(X))])
            train_mse = np.mean((y - curr_preds)**2)
            self.loss_history.append(train_mse)
           
            val_mse = train_mse 
            if X_val is not None:
                val_preds = np.array([[n.feed_neuron(X_val[k]) for n in self.output_neurons] for k in range(len(X_val))])
                val_mse = np.mean((y_val - val_preds)**2)
                self.val_loss_history.append(val_mse)
        
            if X_test is not None:
                test_preds = np.array([[n.feed_neuron(X_test[k]) for n in self.output_neurons] for k in range(len(X_test))])
                test_mse = np.mean((y_test - test_preds)**2)
                self.test_loss_history.append(test_mse)
            
            if val_mse < best_loss - 1e-6:
                best_loss = val_mse
                best_w = [n.weights.copy() for n in self.output_neurons]
                best_b = [n.bias for n in self.output_neurons]
                patience_cnt = 0
            else:
                patience_cnt += 1
                
            if patience_cnt >= patience:
                for j, n in enumerate(self.output_neurons):
                    n.weights = best_w[j]
                    n.bias = best_b[j]
                break
                
            if train_mse < tolerance:
                break

    def _train_candidates(self, X, residuals, n_candidates=8, epochs=100):
        """Allena pool di candidati per massimizzare la correlazione con l'errore residuo."""
        input_dim = X.shape[1]
        
        # Crea candidati
        candidates = []
        for _ in range(n_candidates):
            cand = nrn.Neuron(num_inputs=input_dim, index_in_layer=0, is_output_neuron=False, activation_function_type="tanh")
            cand.weights = np.random.uniform(-1, 1, size=input_dim) 
            candidates.append(cand)
            
        lr_cand = self.learning_rate 
        
        res_mean = np.mean(residuals, axis=0)
        
        for epoch in range(epochs):
            for cand in candidates:
                values = np.array([cand.feed_neuron(x) for x in X])
                
                # Calcola Correlazione S
                v_mean = np.mean(values)
                cand_diff = values - v_mean       
                res_diff = residuals - res_mean   
                
                cov = np.sum(cand_diff[:, None] * res_diff, axis=0) 
                signs = np.sign(cov)
                
                error_term = np.sum(signs * res_diff, axis=1) 
                
                derivs = 1.0 - values**2 
                
                delta = error_term * derivs
                
                grad_w = np.dot(delta, X) / len(X)
                grad_b = np.mean(delta)
                
                cand.weights += lr_cand * grad_w
                cand.bias += lr_cand * grad_b

        # Selezione vincitore
        best_cand = None
        best_score = -1.0
        
        for cand in candidates:
            values = np.array([cand.feed_neuron(x) for x in X])
            v_mean = np.mean(values)
            res_diff = residuals - res_mean
            cov = np.sum((values - v_mean)[:, None] * res_diff, axis=0)
            score = np.sum(np.abs(cov))
            
            if score > best_score:
                best_score = score
                best_cand = cand
                
        print(f"Miglior Candidato Score: {best_score:.4f}")
        return best_cand