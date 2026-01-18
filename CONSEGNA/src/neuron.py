import numpy as np
import math

class Neuron:
    # Constructor
    def __init__(self, num_inputs, index_in_layer, is_output_neuron=False, activation_function_type="tanh", weight_initializer='xavier', weight_floor=-0.01, weight_ceiling=0.01):
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
        self.best_weights = None
        self.best_bias = None
        self.weight_floor = weight_floor
        self.weight_ceiling = weight_ceiling

        limit = 1 / math.sqrt(num_inputs) if num_inputs > 0 else 0.2

        if weight_initializer == 'xavier':
            self.weights = np.random.uniform(-limit, limit, size=num_inputs)
            self.bias = np.random.uniform(-limit, limit)
        else:
            self.weights = np.random.uniform(self.weight_floor, self.weight_ceiling, size=num_inputs)
            self.bias = np.random.uniform(0.0, 0.0)

        # for mini-batch gradient accumulation
        self.prev_weight_grad = np.zeros(num_inputs)
        self.prev_bias_grad = 0.0
        
        # for quickprop
        self.prev_weight_update = np.zeros(num_inputs)
        self.prev_bias_update = 0.0
        
        # for rprop
        self.rprop_step_w = np.full(num_inputs, 0.1) # Serve un valore del passo specifico per ogni peso (dal momento che non usa un eta globale) che viene inizializzato a 0.1
        self.rprop_step_b = 0.1 # Lo stesso per il bias
    
        # for mini-batch gradient accumulation
        self.weight_grad_accum = np.zeros(num_inputs)
        self.bias_grad_accum = 0.0
        
    # Updates the list of output neurons
    def attach_to_output(self, neurons):
        self.attached_neurons = list(neurons)
    
    # activation function gets called dynamically (string based) based on how the neuron was initialized
    def activation_funct(self, input):
        if self.activation_function_type == "sigmoid":
            input = np.clip(input, -500, 500) 
            return(1.0/(1.0 + np.exp(-input)))
        elif self.activation_function_type == "tanh":
            return np.tanh(input)
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
    
    # activation function's derivative gets called dynamically (string based) based on how the neuron was initialized
    def activation_deriv(self, input):
        if self.activation_function_type == "sigmoid":
            sig = self.output 
            deriv = sig * (1 - sig)
        elif self.activation_function_type == "tanh":
             tanh = self.output
             deriv = 1.0 - tanh**2
        else:
            raise ValueError(f"The specified activation function {self.activation_function_type} is not implemented as of yet.")
       
        return np.clip(deriv, 1e-8, 1.0)

    # a single example is fed to the neuron. The sum and then whatever activation function was selected are called
    def feed_neuron(self, inputs):
        if inputs is None:
            print(f"ERROR: inputs is None! Neuron: layer index {self.index_in_layer}, is_output: {self.is_output_neuron}")
            raise ValueError("Inputs cannot be None")

        self.inputs = np.asarray(inputs, dtype=float).flatten()

        if self.inputs.shape[0] != self.weights.shape[0]:
            print(f"DEBUG feed_neuron: inputs shape {self.inputs.shape}, weights shape {self.weights.shape}")
            print(f"  Neuron: layer index {self.index_in_layer}, is_output: {self.is_output_neuron}")
    
        self.net = float(np.dot(self.weights, self.inputs)) + self.bias
        self.output = self.activation_funct(self.net)
        return self.output
    
    # the delta computation for an output neuron and a hidden neuron are different, but
    # this method compounds them, checking the is_output_neuron flag (set at __init__ time)
    def compute_delta(self, signal_error=None):
        if self.is_output_neuron:
            if isinstance(signal_error, np.ndarray):
                signal_error = float(signal_error.flatten()[0])
            else:
                signal_error = float(signal_error)
            self.delta = signal_error * self.activation_deriv(self.output)
        else:
            delta_sum = 0.0
            for k in self.attached_neurons:
                try:
                    weights = np.asarray(k.weights)
                    if weights.ndim == 0:
                        w_kj = float(weights)
                    else:
                        w_kj = float(weights[self.index_in_layer])
                    delta_sum += k.delta * w_kj
                except (IndexError, TypeError) as e:
                    print(f"Warning: Error accessing weight in compute_delta: {e}")
                    continue
            self.delta = delta_sum * self.activation_deriv(self.output)
        return self.delta
    
    def accumulate_gradients(self):
        inputs_array = np.asarray(self.inputs, dtype=float)
        self.weight_grad_accum += self.delta * inputs_array
        self.bias_grad_accum += self.delta
    
    def reset_grad_accum(self):
        self.weight_grad_accum.fill(0.0)
        self.bias_grad_accum = 0.0
    
    def apply_accumulated_gradients(self, **kwargs):
        eta = kwargs.get("eta", 0.1)
        batch_size = kwargs.get("batch_size", 1)
        algorithm = kwargs.get("algorithm", 'sgd')
    
        if algorithm == 'sgd': 
            l2_lambda = kwargs.get('l2_lambda', 0.0)
            momentum = kwargs.get('momentum', 0.0)
            grad_w = self.weight_grad_accum / batch_size
            grad_b = self.bias_grad_accum / batch_size  

        if momentum > 0.0:
            # Accumula velocità con momentum
            self.vel_w = momentum * self.vel_w + grad_w
            self.vel_b = momentum * self.vel_b + grad_b
            # CORREZIONE: Sottrai invece di aggiungere (gradient DESCENT)
            self.weights -= eta * self.vel_w  # ← CAMBIATO DA += A -=
            self.bias -= eta * self.vel_b      # ← CAMBIATO DA += A -=
        else: 
            self.weights -= eta * grad_w
            self.bias -= eta * grad_b
        
        # Applica L2 regularization
        if l2_lambda > 0.0:
            self.weights -= eta * l2_lambda * self.weights

        elif algorithm == 'rprop':
            eta_plus = kwargs.get('eta_plus', 1.2)
            eta_minus = kwargs.get('eta_minus', 0.5)
            delta_min = kwargs.get('delta_min', 1e-6)
            delta_max = kwargs.get('delta_max', 50.0)
            l2_lambda = kwargs.get('l2_lambda', 0.0)
        
            self.update_weights_rprop(batch_size, eta_plus, eta_minus, delta_min, delta_max, l2_lambda)
            self.reset_grad_accum() 
    
        elif algorithm == 'quickprop':
            mu = kwargs.get('mu', 1.75)
            decay = kwargs.get('decay', -0.0001)
        
            self.update_weights_quickprop(batch_size, eta, mu, decay)
            self.reset_grad_accum()
    
        # Reset accumulatori
        self.weight_grad_accum.fill(0.0)
        self.bias_grad_accum = 0.0

        if algorithm == 'rprop':
            eta_plus = kwargs.get('eta_plus', 1.2)
            eta_minus = kwargs.get('eta_minus', 0.5)
            delta_min = kwargs.get('delta_min', 1e-6)
            delta_max = kwargs.get('delta_max', 50.0)
            l2_lambda = kwargs.get('l2_lambda', 0.0)
            
            self.update_weights_rprop(batch_size, eta_plus, eta_minus, delta_min, delta_max, l2_lambda)
            self.reset_grad_accum() 
        
        elif algorithm == 'quickprop':
            mu = kwargs.get('mu', 1.75)
            decay = kwargs.get('decay', -0.0001)
            
            self.update_weights_quickprop(batch_size, eta, mu, decay)
            self.reset_grad_accum()
        
        self.weight_grad_accum.fill(0.0)
        self.bias_grad_accum = 0.0

    def update_weights(self, eta, l2_lambda=0.00):
        if self.inputs is None:
            self.inputs = np.zeros(self.weights.shape[0])
    
        inputs_array = np.asarray(self.inputs, dtype=float).flatten()
    
        if inputs_array.shape[0] != self.weights.shape[0]:
            self.inputs = np.zeros(self.weights.shape[0])
            inputs_array = self.inputs
    
        weight_update = self.delta * inputs_array - l2_lambda * self.weights
        self.weights -= eta * weight_update
        self.bias -= eta * self.delta

        if l2_lambda > 0.0:
            self.weights -= eta * l2_lambda * self.weights

    def set_best_weights(self):
        self.best_weights = self.weights
        self.best_bias = self.bias

    def restore_best_weights(self):
        self.weights = self.best_weights
        self.bias = self.best_bias
    
    def update_weights_rprop(self, batch_size, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50.0, l2_lambda=0.0):
        curr_grad_w = self.weight_grad_accum / batch_size
        curr_grad_b = self.bias_grad_accum / batch_size

        if l2_lambda > 0.0:
            curr_grad_w -= l2_lambda * self.weights
             
        for i in range(len(self.weights)):
            change = self.prev_weight_grad[i] * curr_grad_w[i]

            if change > 0: 
                self.rprop_step_w[i] = min(self.rprop_step_w[i] * eta_plus, delta_max) 
                weight_delta = np.sign(curr_grad_w[i]) * self.rprop_step_w[i] 
                
                self.weights[i] += weight_delta 
                self.prev_weight_grad[i] = curr_grad_w[i]
                self.prev_weight_update[i] = weight_delta 

            elif change < 0: 
                self.rprop_step_w[i] = max(self.rprop_step_w[i] * eta_minus, delta_min) 
                self.weights[i] -= self.prev_weight_update[i] 
                self.prev_weight_grad[i] = 0 
                
            else: 
                weight_delta = np.sign(curr_grad_w[i]) * self.rprop_step_w[i] 
                self.weights[i] -= weight_delta
                self.prev_weight_grad[i] = curr_grad_w[i]
                self.prev_weight_update[i] = weight_delta

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
        curr_grad_w = self.weight_grad_accum / batch_size
        curr_grad_b = self.bias_grad_accum / batch_size

        for i in range(len(self.weights)):
            grad_descent = curr_grad_w[i]
            current_slope = -grad_descent + (decay * self.weights[i]) 
            prev_slope = self.prev_weight_grad[i]
            prev_step= self.prev_weight_update[i]
            
            step = 0.0
            # Ignition
            if abs(prev_step) < 1e-10: 
                step = - eta * current_slope 
            else:
                grad_diff = prev_slope - current_slope
                if abs(grad_diff) < 1e-10:
                    step = - eta * current_slope
                else:
                    step = (current_slope / grad_diff) * prev_step
                    max_step = mu * abs(prev_step) 
                    if abs(step) > max_step:
                        step = max_step*np.sign(step) 
                    if np.sign(step) == np.sign(current_slope):
                        step = - eta * current_slope 
                        
            self.weights[i] += step 
            self.prev_weight_grad[i] = current_slope 
            self.prev_weight_update[i] = step
    
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
        
        self.bias += step_b 
        self.prev_bias_grad = curr_grad_b
        self.prev_bias_update = step_b 