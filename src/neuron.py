import numpy as np
import math

class Neuron:
    # Constructor
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False, activation_function_type="tanh", weight_initializer='xavier', bias_initializer='uniform'):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron
        self.activation_function_type = activation_function_type
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.inputs = []
        self.attached_neurons = []
        self.vel_w = np.zeros(num_inputs)
        self.vel_b = 0.0

        limit = 1 / math.sqrt(num_inputs) if num_inputs > 0 else 0.2

        if weight_initializer == 'xavier':
            self.weights = np.random.uniform(-limit, limit, size=num_inputs)
            self.bias = 0.0 
        else:
            self.weights = np.random.uniform(-0.2, 0.2, size=num_inputs)
            self.bias = np.random.uniform(-0.2, 0.2)

        #per mini-batch gradient accumulation
        self.prev_weight_grad = np.zeros(num_inputs) # Per ricordare il gradiente dei pesi al passo precedente (t-1)
        self.prev_bias_grad = 0.0 # Inizializza a zero la memoria del gradiente del bias al passo precedente 
        
        # Per quickprop
        self.prev_weight_update = np.zeros(num_inputs) # Per ricordare l'ultimo aggiornamento dei pesi (t-1)
        self.prev_bias_update = 0.0 # Inizializza a zero la memoria dell'ultimo aggiornamento del bias
        
        # Per rprop
        self.rprop_step_w = np.full(num_inputs, 0.1) # Serve un valore del passo specifico per ogni peso (dal momento che non usa un eta globale) che viene inizializzato a 0.1
        self.rprop_step_b = 0.1 # Lo stesso per il bias
    
        # Per mini-batch gradient accumulation
        self.weight_grad_accum = np.zeros(num_inputs)
        self.bias_grad_accum = 0.0
        
    # Updates the list of output neurons
    def attach_to_output(self, neurons):
        self.attached_neurons = list(neurons)
    
    # Activation function gets called dynamically (string based) based on how the neuron was initialized
    def activation_funct(self, input):
        if self.activation_function_type == "sigmoid":
            input = np.clip(input, -500, 500)  # Evita overflow
            return(1.0/(1.0 + np.exp(-input)))
        elif self.activation_function_type == "tanh":
            return np.tanh(input)
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
    
    # Activation function's derivative gets called dynamically (string based) based on how the neuron was initialized
    def activation_deriv(self, input):
        if self.activation_function_type == "sigmoid":
            deriv = input * (1 - input)
        elif self.activation_function_type == "tanh":
            deriv = 1.0 - np.square(input)
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
        return np.maximum(deriv, 1e-8)

    # a single example is fed to the neuron. The sum and then whatever activation function was selected are called
    def feed_neuron(self, inputs):
        self.inputs = np.array(inputs, dtype=float)
        self.net = float(np.dot(self.weights, inputs)) + self.bias
        self.output = self.activation_funct(self.net)
        return self.output
    
    # The delta computation for an output neuron and a hidden neuron are different, but
    # this method compounds them, checking the is_output_neuron flag (set at __init__ time)
    def compute_delta(self, signal_error=None): # target viene sostituito con signal_error che arriva direttamente da neural_network.backward
        if self.is_output_neuron:
            self.delta = signal_error * self.activation_deriv(self.output)
        else:
            delta_sum = 0.0
            for k in self.attached_neurons:
                w_kj = k.weights[self.index_in_layer]
                delta_sum += k.delta * w_kj
            self.delta = delta_sum * self.activation_deriv(self.output)
        return self.delta
    
    def accumulate_gradients(self):
        """Accumula gradienti invece di aggiornare immediatamente"""
        self.weight_grad_accum += self.delta * self.inputs
        self.bias_grad_accum += self.delta
    
    def reset_grad_accum(self):
        """Resetta accumulatore gradienti"""
        self.weight_grad_accum.fill(0.0)
        self.bias_grad_accum = 0.0
    
    def apply_accumulated_gradients(self, **kwargs):
        
        """Applica gradienti accumulati (media del batch) a seconda dell'algoritmo scelto"""
        eta = kwargs.get("eta", 0.1)
        batch_size = kwargs.get("batch_size", 1)
        algorithm = kwargs.get("algorithm", 'sgd')
        
        if algorithm == 'sgd':
            l2_lambda = kwargs.get('l2_lambda', 0.0)
            momentum = kwargs.get('momentum', 0.0)
            # media del gradiente sul batch
            grad_w = self.weight_grad_accum / batch_size
            grad_b = self.bias_grad_accum / batch_size  

            if momentum >= 0.0:
                self.vel_w = momentum * self.vel_w + grad_w
                self.vel_b = momentum * self.vel_b + grad_b
                self.weights -= eta * self.vel_w
                self.bias -= eta * self.vel_b  
            else:
             self.weights -= eta * grad_w
            self.bias -= eta * grad_b  
            # Reset accumulatori batch
            self.weight_grad_accum.fill(0.0)
            self.bias_grad_accum = 0.0 
        
        elif algorithm == 'rprop':
            eta_plus = kwargs.get('eta_plus', 1.2)
            eta_minus = kwargs.get('eta_minus', 0.5)
            delta_min = kwargs.get('delta_min', 1e-6)
            delta_max = kwargs.get('delta_max', 50.0)
            l2_lambda = kwargs.get('l2_lambda', 0.0)
            
            self.update_weights_rprop(batch_size, eta_plus, eta_minus, delta_min, delta_max, l2_lambda)
            self.reset_grad_accum() # Resetta gli accumuli dell'aggiornamento dei pesi per il prossimo batch
        
        elif algorithm == 'quickprop':
            mu = kwargs.get('mu', 1.75)
            decay = kwargs.get('decay', -0.0001)
            
            self.update_weights_quickprop(batch_size, eta, mu, decay)
            self.reset_grad_accum() # Resetta gli accumuli dell'aggiornamento dei pesi per il prossimo batch

    def update_weights(self, eta, l2_lambda=0.00):
        self.weights -= eta * (self.delta * self.inputs - l2_lambda * self.weights)
        self.bias -= eta * self.delta
    
    def update_weights_rprop(self, batch_size, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50.0, l2_lambda=0.0):
        """
        Implementazione dell'algoritmo RPROP per l'aggiornamento dei pesi.
        """
        # Normalizzazione del gradiente per evitare dipendenza dalla dimensione del batch
        curr_grad_w = self.weight_grad_accum / batch_size
        curr_grad_b = self.bias_grad_accum / batch_size

        if l2_lambda > 0.0: # Penalizzazione dei pesi grandi tramite L2 regularization
            curr_grad_w -= l2_lambda * self.weights
            
        # Aggiornamento pesi 
        for i in range(len(self.weights)):
            # Prodotto dei gradienti (t-1) * (t)
            change = self.prev_weight_grad[i] * curr_grad_w[i]

            if change > 0: # Il gradiente ha mantenuto lo stesso segno rispetto al passo precedente 
                self.rprop_step_w[i] = min(self.rprop_step_w[i] * eta_plus, delta_max) # Aumento del passo poiché si sta andando nella direzione giusta
                weight_delta = np.sign(curr_grad_w[i]) * self.rprop_step_w[i] # Si prende solo il segno del gradiente corrente e lo si aggiunge al passo aggiornato
                
                self.weights[i] += weight_delta # Aggiornamento effettivo del peso 
                self.prev_weight_grad[i] = curr_grad_w[i] # Memorizzazione del gradiente corrente per il prossimo confronto perchè questo valore al prossimo passo diventerà t-1
                self.prev_weight_update[i] = weight_delta # Memorizzazione dell'ultimo aggiornamento del peso

            elif change < 0: # Il gradiente ha cambiato segno rispetto al passo precedente (sta andando nella direzione opposta)
                self.rprop_step_w[i] = max(self.rprop_step_w[i] * eta_minus, delta_min) # Diminuzione del passo poiché si è oltrepassato il minimo locale
                self.weights[i] -= self.prev_weight_update[i] # Si annulla l'ultimo aggiornamento del peso (=si torna al passo precedente) perchè troppo grande
                self.prev_weight_grad[i] = 0 # Il gradiente viene ripristinato a zero per evitare di fare un ulteriore aggiornamento in questa direzione al prossimo passo (backtracking)
                
            else: # Il gradiente è zero o non ha cambiato segno (all'inizio del tr abbiamo t= 0 quindi non esiste un gradiente precedente perchè il prodotto è 0)
                weight_delta = np.sign(curr_grad_w[i]) * self.rprop_step_w[i] # Si prende solo il segno del gradiente corrente e lo si aggiunge al passo attuale
                self.weights[i] -= weight_delta
                # Ricostruzione della memoria
                self.prev_weight_grad[i] = curr_grad_w[i]
                self.prev_weight_update[i] = weight_delta

        # Aggiornamento bias con lo stesso metodo
        change_b = self.prev_bias_grad * curr_grad_b
        
        if change_b > 0:
            self.rprop_step_b = min(self.rprop_step_b * eta_plus, delta_max)
            bias_delta = np.sign(curr_grad_b) * self.rprop_step_b
            self.bias -= bias_delta
            self.prev_bias_grad = curr_grad_b
            self.prev_bias_update = bias_delta
            
        elif change_b < 0:
            self.rprop_step_b = max(self.rprop_step_b * eta_minus, delta_min)
            self.bias -= self.prev_bias_update 
            self.prev_bias_grad = 0 
            
        else:
            bias_delta = np.sign(curr_grad_b) * self.rprop_step_b
            self.bias -= bias_delta
            self.prev_bias_grad = curr_grad_b
            self.prev_bias_update = bias_delta
    
    def update_weights_quickprop(self, batch_size, eta, mu=1.75, decay=-0.0001):
        """
        Implementazione dell'algoritmo Quickprop per l'aggiornamento dei pesi.
        """
        # Normalizzazione del gradiente per evitare dipendenza dalla dimensione del batch
        curr_grad_w = self.weight_grad_accum / batch_size
        curr_grad_b = self.bias_grad_accum / batch_size

        # Aggiornamento pesi 
        for i in range(len(self.weights)):
            grad_descent = curr_grad_w[i]
            current_slope = -grad_descent + (decay * self.weights[i]) # Aggiunta del termine di weight decay per evitare che i pesi facciano passi troppo grandi
            prev_slope = self.prev_weight_grad[i]
            prev_step= self.prev_weight_update[i]
            
            step = 0.0
            # Ignition
            if abs(prev_step) < 1e-10: # Se il passo precedente è molto piccolo (ossia siamo all'inizio del training)
                step = - eta * current_slope # Si usa la discesa del gradiente standard per generare il primo passo per arrivare al secondo punto (il primo è l'inizializzazione)
            else:
                grad_diff = prev_slope - current_slope
                if abs(grad_diff) < 1e-10:
                    step = - eta * current_slope
                else:
                    step = (current_slope / grad_diff) * prev_step # Calcolo del passo quickprop!!
                    max_step = mu * abs(prev_step) # Calcolo del passo massimo consentito per evitare salti troppo grandi
                    if abs(step) > max_step:
                        step = max_step*np.sign(step) # Ignoriamo la grandezza suggerita dalla formula e impostiamo la grandezza al limite massimo consentito mantenendo la direzione originale (sign)
                    if np.sign(step) == np.sign(current_slope): # Se il passo calcolato ha la stessa direzione del gradiente (ossia si va nella direzione opposta alla discesa del gradiente)
                        step = - eta * current_slope # Si forza l'aggiornamento a seguire la discesa del gradiente
                        
            self.weights[i] -= step # Aggiornamento effettivo del peso
            self.prev_weight_grad[i] = current_slope # Memorizzazione del gradiente corrente per il prossimo confronto perchè questo valore al prossimo passo diventerà t-1
            self.prev_weight_update[i] = step # Memorizzazione dell'ultimo aggiornamento del peso
    
        # Aggiornamento bias con lo stesso metodo
        current_slope_b = - curr_grad_b 
        prev_slope_b = self.prev_bias_grad
        
        step_b = 0.0
        if abs(self.prev_bias_update) < 1e-10:
            step_b = - eta * current_slope_b
        else:
            slope_diff_b = prev_slope_b - current_slope_b
            if abs(slope_diff_b) < 1e-10:
                step_b = - eta * current_slope_b
            else:
                step_b = (current_slope_b / slope_diff_b) * self.prev_bias_update
                if abs(step_b) > mu * abs(self.prev_bias_update):
                    step_b = mu * abs(self.prev_bias_update) * np.sign(step_b)
                if np.sign(step_b) == np.sign(current_slope_b):
                    step_b = - eta * current_slope_b
        
        self.bias -= step_b # Aggiornamento effettivo del bias
        # Ricostruzione della memoria
        self.prev_bias_grad = curr_grad_b
        self.prev_bias_update = step_b 