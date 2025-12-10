import neuron
import numpy as np
import matplotlib.pyplot as plt
import utils

class NeuralNetwork:
    # Constructor
    # network_structure is the number of neurons in input, hidden layers
    # and output, expressed as an array. For example [6, 2, 2, 1].
    def __init__(self, network_structure, eta=0.1, loss_type="half_mse", l2_lambda=0.00): # Modificato il costruttore per accettare diversi tipi di loss
        self.eta = eta
        self.l2_lambda = l2_lambda
        self.loss_history = {"training": [], "test": []}
        self.loss_type = loss_type

        self.layers = []
        self.layers.append([neuron.Neuron(num_inputs=0, index_in_layer=j, 
                activation_function_type="sigmoid", is_output_neuron=False) 
                for j in range(network_structure[0])])
        
        for i in range(len(network_structure) - 1):
            self.layers.append([neuron.Neuron(num_inputs=network_structure[i], 
                        index_in_layer=j, activation_function_type="sigmoid", 
                        is_output_neuron=(i==len(network_structure)-2)) 
                        for j in range(network_structure[i + 1])])

        for l in range(len(self.layers) - 1):
            for n in self.layers[l]:
                n.attach_to_output(self.layers[l + 1])

    # a single example is propagated through the network, calling
    # the feed_neuron method for each neuron
    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = [n.feed_neuron(x) for n in self.layers[l + 1]]
        return x

    # streamlines and encapsulates the forwarding of multiple examples
    def predict(self, X):
        preds = []
        for xi in X:
            out = self.forward(xi)
            if isinstance(out, (list, np.ndarray)):
                preds.append(float(out[0]))
            else:
                preds.append(float(out))
        return np.array(preds)

    def _reset_gradients(self):
        """Resetta tutti gli accumulatori di gradienti"""
        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                neuron.reset_grad_accum()

    def _apply_accumulated_gradients(self, batch_size):
        """Applica gradienti accumulati a tutti i neuroni"""
        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                neuron.apply_accumulated_gradients(self.eta, batch_size, l2_lambda=self.l2_lambda)

    # backprop implementation
    def backward(self, error_signals, accumulate=False):
        """Versione modificata per supportare accumulazione"""
        if np.isscalar(error_signals) or (isinstance(error_signals, np.ndarray) and error_signals.ndim == 0):
            error_signals = [error_signals]
        
        for i, neuron in enumerate(self.layers[-1]):
            neuron.compute_delta(error_signals[i])
        
        for l in range(len(self.layers) - 2, 0, -1):
            for neuron in self.layers[l]:
                neuron.compute_delta(None)

        for l in range(len(self.layers) - 1, 0, -1):
            for n in self.layers[l]:
                if accumulate:
                    n.accumulate_gradients()
                else:
                    n.update_weights(self.eta, l2_lambda=self.l2_lambda)
                    
    # core training method.
    # the test set is passed purely to assess the test error at each step but is not used for
    # learning, to keep the test set "unseen".
    # Nothing is returned because the network's weights are updated in place. (we choose to have a stateful network)
    def fit(self, X, X_test, y, y_test, epochs=1000, batch_size=1, verbose=True):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        for epoch in range(epochs):
            total_loss = 0.0
            test_loss = 0.0
            
            if batch_size == 1:  # Online learning (comportamento originale)
                for xi, yi in zip(X, y):
                    outputs = self.forward(xi)
                    y_pred = outputs[0]
                    total_loss += self.compute_loss(yi, y_pred, loss_type=self.loss_type)
                    error_signal = self.compute_error_signal(yi, y_pred, loss_type=self.loss_type)
                    self.backward(error_signal)
            else:  # Mini-batch con shuffling
                # 1. Shuffle dei dati all'inizio di ogni epoca
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                # 2. Divisione in batch
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    
                    # Reset accumulazione gradienti per ogni batch
                    self._reset_gradients()
                    
                    # Accumula gradienti per tutti i campioni nel batch
                    batch_loss = 0.0
                    for i in range(start_idx, end_idx):
                        xi = X_shuffled[i]
                        yi = y_shuffled[i]
                        
                        outputs = self.forward(xi)
                        y_pred = outputs[0]
                        batch_loss += self.compute_loss(yi, y_pred, loss_type=self.loss_type)
                        error_signal = self.compute_error_signal(yi, y_pred, loss_type=self.loss_type)
                        self.backward(error_signal, accumulate=True)
                    
                    # Applica gradienti accumulati (media del batch)
                    self._apply_accumulated_gradients(batch_size=end_idx - start_idx)
                    total_loss += batch_loss
            
            avg_loss = total_loss / len(X)
            self.loss_history["training"].append(avg_loss)
            
            y_pred_test = self.predict(X_test)
            test_loss = np.sum(self.compute_loss(y_test, y_pred_test.flatten(), loss_type=self.loss_type))
            avg_test_loss = test_loss / len(y_test)
            self.loss_history["test"].append(avg_test_loss)
            
            if epoch % 25 == 0 and verbose:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Batch size: {batch_size}")

    # a method to save the losses of the training and test sets as a plot
    def save_plots(self, path):
        plt.plot(self.loss_history["training"], label='Training Loss')
        plt.plot(self.loss_history["test"], label='Test Loss')
        plt.legend()
        plt.ylim(0, max(max(self.loss_history["training"]), max(self.loss_history["test"])) * 1.1)
        plt.savefig(path)

    def draw_network(self, path):
        utils.draw_network(self.layers, path)
        
    def compute_loss(self, y_true, y_pred, loss_type="half_mse"):
        """
        Calcola la loss in base al tipo specificato.
        """
        if loss_type == "half_mse":
            return 0.5 * (y_true - y_pred) ** 2
        elif loss_type == "mae":
            return np.abs(y_true - y_pred)
        elif loss_type == "log_cosh":
            return np.log(np.cosh(y_pred - y_true))
        elif loss_type == "huber":
            delta = 1.0
            error = y_true - y_pred
            is_small_error = np.abs(error) <= delta
            squared_loss = 0.5 * (error ** 2)
            linear_loss = delta * (np.abs(error) - 0.5 * delta)
            return np.where(is_small_error, squared_loss, linear_loss)
        else:
            raise ValueError(f"Loss type '{loss_type}' not implemented.")
    
    def compute_error_signal(self, y_true, y_pred, loss_type):
        """
        Calcola il segnale di errore da passare al neurone di output in base al tipo di loss specificato
        """
        error = y_true - y_pred 
        
        if loss_type == "half_mse":
            return error
        
        elif loss_type == "mae":
            return np.sign(error)
        
        elif loss_type == "log_cosh":
            return np.tanh(error)
        
        elif loss_type == "huber":
            delta = 1.0 # Deve coincidere con quello usato in compute_loss
            is_small_error = np.abs(error) <= delta
            return np.where(is_small_error, error, delta * np.sign(error)) # Se l'errore è piccolo si comporta come MSE (error), altrimenti come MAE (delta * sign(error))
        else:
            raise ValueError(f"Loss type '{loss_type}' not implemented.")
    
    

   