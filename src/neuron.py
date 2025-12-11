import numpy as np
import math

class Neuron:
    # Constructor
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False, activation_function_type="tanh", weight_initializer='xavier', bias_initializer='uniform'):
        self.index_in_layer = index_in_layer
        self.is_output_neuron = is_output_neuron
        self.weights = np.random.uniform(-0.5, 0.5, size=num_inputs)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.activation_function_type = activation_function_type
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.inputs = []
        self.attached_neurons = []

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
            return(1/(1 + np.exp(-input)))
        elif self.activation_function_type == "tanh":
            return((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)))
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
    
    # Activation function's derivative gets called dynamically (string based) based on how the neuron was initialized
    def activation_deriv(self, input):
        if self.activation_function_type == "sigmoid":
            return input * (1 - input) + 0.1 # Aggiunta del valore 0.1 per evitare che l'algoritmo si blocchi in caso di derivata nulla
        elif self.activation_function_type == "tanh":
            return 1 - np.square(input) + 0.1
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")

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
    
    def apply_accumulated_gradients(self, eta, batch_size, algorithm='sgd', **kwargs):
        
        """Applica gradienti accumulati (media del batch) a seconda dell'algoritmo scelto"""
        
        if algorithm == 'sgd':
            l2_lambda = kwargs.get('l2_lambda', 0.0)
            grad_w = self.weight_grad_accum / batch_size
            grad_b = self.bias_grad_accum / batch_size         
            self.weights += eta * (grad_w - l2_lambda * self.weights)
            self.bias += eta * grad_b
            # Reset per SGD
            self.weight_grad_accum.fill(0.0)
            self.bias_grad_accum = 0.0
        
        elif algorithm == 'rprop':
            eta_plus = kwargs.get('eta_plus', 1.2)
            eta_minus = kwargs.get('eta_minus', 0.5)
            delta_min = kwargs.get('delta_min', 1e-6)
            delta_max = kwargs.get('delta_max', 50.0)
            
            self.update_weights_rprop(batch_size, eta_plus, eta_minus, delta_min, delta_max)
            self.reset_grad_accum() # Resetta gli accumuli dell'aggiornamento dei pesi per il prossimo batch
    

    def update_weights(self, eta, l2_lambda=0.00):
        self.weights += eta * (self.delta * self.inputs - l2_lambda * self.weights)
        self.bias += eta * self.delta
    
    def update_weights_rprop(self, batch_size, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50.0):
        """
        Implementazione dell'algoritmo RPROP per l'aggiornamento dei pesi.
        """
        # Normalizzazione del gradiente per evitare dipendenza dalla dimensione del batch
        curr_grad_w = self.weight_grad_accum / batch_size
        curr_grad_b = self.bias_grad_accum / batch_size

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
                self.weights[i] += weight_delta
                # Ricostruzione della memoria
                self.prev_weight_grad[i] = curr_grad_w[i]
                self.prev_weight_update[i] = weight_delta

        # Aggiornamento bias con lo stesso metodo
        change_b = self.prev_bias_grad * curr_grad_b
        
        if change_b > 0:
            self.rprop_step_b = min(self.rprop_step_b * eta_plus, delta_max)
            bias_delta = np.sign(curr_grad_b) * self.rprop_step_b
            self.bias += bias_delta
            self.prev_bias_grad = curr_grad_b
            self.prev_bias_update = bias_delta
            
        elif change_b < 0:
            self.rprop_step_b = max(self.rprop_step_b * eta_minus, delta_min)
            self.bias -= self.prev_bias_update 
            self.prev_bias_grad = 0 
            
        else:
            bias_delta = np.sign(curr_grad_b) * self.rprop_step_b
            self.bias += bias_delta
            self.prev_bias_grad = curr_grad_b
            self.prev_bias_update = bias_delta