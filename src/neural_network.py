import neuron
import numpy as np
import matplotlib.pyplot as plt
import utils

class NeuralNetwork:
    # Constructor
    # network_structure is the number of neurons in input, hidden layers
    # and output, expressed as an array. For example [6, 2, 2, 1].
    def __init__(self, network_structure, **kwargs):
        self.eta = kwargs.get("eta", 0.1)
        self.loss_type = kwargs.get("loss_type", "half_mse")
        self.l2_lambda = kwargs.get("l2_lambda", 0.00)
        self.algorithm = kwargs.get("algorithm", "sgd")
        self.activation_type = kwargs.get("activation_type", "sigmoid")
        self.eta_plus = kwargs.get("eta_plus", 1.2)
        self.eta_minus = kwargs.get("eta_minus", 0.5)
        self.weight_initializer = kwargs.get("weight_initializer", "def")
        self.mu = kwargs.get("mu", 1.75)
        self.decay = kwargs.get("decay", 0.001)
        self.loss_history = {"training": [], "validation": []}
        self.layers = []
        self.momentum = kwargs.get("momentum", 0.0) # Implementazione momentum

        self.layers.append([neuron.Neuron(num_inputs=0, index_in_layer=j, 
                activation_function_type=self.activation_type, is_output_neuron=False, weight_initializer=self.weight_initializer) 
                for j in range(network_structure[0])])
        
        for i in range(len(network_structure) - 1):
            self.layers.append([neuron.Neuron(num_inputs=network_structure[i], 
                        index_in_layer=j, activation_function_type=self.activation_type, 
                        is_output_neuron=(i==len(network_structure)-2), weight_initializer=self.weight_initializer) 
                        for j in range(network_structure[i + 1])])

        for l in range(len(self.layers) - 1):
            for n in self.layers[l]:
                n.attach_to_output(self.layers[l + 1])

    # a single example is propagated through the network, calling
    # the feed_neuron method for each neuron
    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = [n.feed_neuron(x) for n in self.layers[l + 1]]
        return np.array(x, dtype="float")

    # streamlines and encapsulates the forwarding of multiple examples
    def predict(self, X):
        preds = [self.forward(xi) for xi in X]
        return np.vstack(preds)

    def _reset_gradients(self):
        """Resetta tutti gli accumulatori di gradienti"""
        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                neuron.reset_grad_accum()

    def _apply_accumulated_gradients(self, batch_size):
        """Applica gradienti accumulati a tutti i neuroni"""
        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                neuron.apply_accumulated_gradients(eta=self.eta, 
                                                   batch_size=batch_size, 
                                                   l2_lambda=self.l2_lambda, 
                                                   algorithm=self.algorithm, 
                                                   eta_plus=self.eta_plus, 
                                                   eta_minus=self.eta_minus,
                                                   momentum=self.momentum)

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
    def fit(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=1, patience=10, verbose=True):

        patience_level = patience

         # reset history
        self.loss_history = {"training": [], "validation": []}

        for epoch in range(epochs):
            total_loss = 0.0

            # Unified batch loop
            for start_idx in range(0, len(X_train), batch_size):

                end_idx = min(start_idx + batch_size, len(X_train))
                current_batch_size = end_idx - start_idx

                # Reset accumulators
                self._reset_gradients()

                batch_loss = 0.0

                # Accumulate gradients inside the batch
                for i in range(start_idx, end_idx):
                    xi = X_train[i]
                    yi = y_train[i]

                    y_pred = self.forward(xi)
                    batch_loss += np.sum(self.compute_loss(yi, y_pred, loss_type=self.loss_type))

                    err = self.compute_error_signal(yi, y_pred, loss_type=self.loss_type)
                    self.backward(err, accumulate=True)

                #apply accumulated gradient once
                self._apply_accumulated_gradients(batch_size=current_batch_size)
                total_loss += batch_loss
            
            #Training loss (media per esempio)
            avg_loss = total_loss / len(X_train)
            self.loss_history["training"].append(avg_loss)

            #Validation loss (non tocca i pesi)
            y_pred_val = self.predict(X_val)
            val_loss = np.sum(self.compute_loss(y_val, y_pred_val, loss_type=self.loss_type))
            avg_val_loss = val_loss / len(y_val)
            self.loss_history["validation"].append(avg_val_loss)

            #early stopping su validation
            if epoch >= patience:
                if self.loss_history["validation"][epoch] >= self.loss_history["validation"][epoch - 1] * 0.99:
                    patience_level -= 1
                else:
                    patience_level = patience
                if patience_level == 0:
                    if verbose:
                        print(f"Early stopping. Last epoch {epoch}, Train Loss: {avg_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Batch size: {batch_size}")
                break       
                    

            if epoch % 25 == 0 and verbose:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Batch size: {batch_size}\nValidation Loss: {avg_val_loss:.4f}\n")
        
    def compute_loss(self, y_true, y_pred, loss_type="half_mse"):
        """
        Calcola la loss in base al tipo specificato.
        """
        error = y_pred - y_true

        if loss_type == "half_mse":
            return 0.5 * error ** 2
        elif loss_type == "mae":
            return np.abs(error)
        elif loss_type == "log_cosh":
            z = y_pred - y_true
            a = np.abs(z)
            return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)
        elif loss_type == "huber":
            delta = 1.0
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
        error = y_pred - y_true
        
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
    
    # a method to save the losses of the training and test sets as a plot
    def save_plots(self, path):
        plt.plot(self.loss_history["training"], label='Training Loss')
        plt.plot(self.loss_history["validation"], label='Validation Loss')
        plt.legend()
        plt.ylim(0, max(max(self.loss_history["training"]), max(self.loss_history["validation"])) * 1.1)
        plt.savefig(path)

    def draw_network(self, path):
        utils.draw_network(self.layers, path)