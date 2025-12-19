import neuron
import numpy as np
import matplotlib.pyplot as plt
import utils
from joblib import Parallel, delayed


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
        self.loss_history = {"training": [], "validation": []}
        self.layers = [] # layer 0: input (vuoto)
        self.layers.append([])
        self.momentum = kwargs.get("momentum", 0.0) # Implementazione momentum
        self.best_val_loss = np.inf
        self.lr_wait = 0
        self.decay = kwargs.get("decay", 0.9)

        self.debug = kwargs.get("debug", False)

        # Creo il primo hidden layer
        if len(network_structure) > 2:  # Se ci sono hidden layers
            # ciclo che scorre sui layer nascosti
            for i in range(1, len(network_structure) - 1):
                # crea il layer i-esimo
                layer_size = network_structure[i]
# quanti neuroni c’erano nello strato precedente (cioè quanti input arrivano a ogni neurone 
# di questo layer)
                prev_layer_size = network_structure[i-1]
# per i=1: layer_size=8, prev_layer_size=17 alloraogni neurone nascosto riceve 17 input
# per i=2: layer_size=8, prev_layer_size=8 allora ogni neurone del secondo hidden riceve 8 input

                #creo una lista vuota dove mettere i neuroni del layer i-esimo
                layer = []
                # ciclo che scorre sui neuroni del layer i-esimo
                for j in range(layer_size):
                    n = neuron.Neuron(
                        num_inputs=prev_layer_size,
                        index_in_layer=j,
                        activation_function_type=self.activation_type,
                        is_output_neuron=False,  # Hidden neurons
                        weight_initializer=self.weight_initializer
                    )
                    #aggiunge il neurone j-esimo al layer i-esimo
                    layer.append(n) 
                    # aggiunge il layer i-esimo alla lista dei layer della rete
                self.layers.append(layer) 
    
        # Creo l'output layer
        output_layer_size = network_structure[-1] #ultimo elemento della lista
        # dimensione del layer precedente
        # l'if serve nel caso in cui non ci siano hidden layer
        prev_layer_size = network_structure[-2] if len(network_structure) > 1 else network_structure[0]

        #lista vuota che conterrà i neuroni dell'output layer
        output_layer = []

        # ciclo che scorre sui neuroni dell'output layer
        for j in range(output_layer_size):
            n = neuron.Neuron(
                num_inputs=prev_layer_size,
                index_in_layer=j,
                activation_function_type=self.activation_type,
                is_output_neuron=True,  # Output neurons
                weight_initializer=self.weight_initializer
            )
            output_layer.append(n)
        # Aggiungo l’output layer alla lista dei layer della rete.
        self.layers.append(output_layer)
    
        # Collego i neuroni (solo se ci sono almeno 2 layer con neuroni)
        if len(self.layers) > 2:
            for l in range(1, len(self.layers) - 1):
                for n in self.layers[l]:
                    n.attach_to_output(self.layers[l + 1])
    # se self.layers ha solo 2 layer (input e output) allora collego direttamente input-output
    # altrimenti collego hidden-hidden e infine hidden-output

   
    def forward(self, x):
       # Propagazione in avanti per un singolo esempio.

        # Se x è un batch (2D), usa predict
        # se x è 1D continua con forward
        if len(x.shape) > 1:
            return self.predict(x)
        
        # forzo x a diventare un array 1D
        x = np.array(x, dtype=float).flatten()
        
        # current values: valori che passano al layer successivo
        current_values = x
        
        # Propaga attraverso tutti i layer con neuroni (partendo da 1, in quanto 0 è input)
        for layer_idx in range(1, len(self.layers)):

            # output di questo layer
            layer_output = []

            for neuron in self.layers[layer_idx]: #per ogni neurone del layer corrente
                # feed_neuron ritorna l'output del neurone
                output = neuron.feed_neuron(current_values)
                #poi salvo quel valore nell'output del layer
                layer_output.append(output)
            
            # l'output di questo layer diventa l'input per il prossimo
            current_values = np.array(layer_output, dtype=float)

        return current_values
    
    # predict fa passare l'input attraverso tutta la rete e ritorna l'output e lo fa
    # per tutti gli input del batch, uno alla volta e restituisce tutte le predizioni insieme
    def predict(self, X):
        #Predizione per un batch di esempi.
        # X: array 2D (n_esempi, n_feature)

        # Se X è un singolo esempio (1D), convertilo in 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # salvo le predizioni per ogni esempio
        predictions = []
        # ciclo su ogni esempio
        for i in range(X.shape[0]):
#chiamo forward per ogni esempio che fa passare l'input attraverso tutti i layer 
# e produce l'output
            pred = self.forward(X[i])
            # salvo la predizione
            predictions.append(pred)
        #infine ritorno tutte le predizioni come array 2D
        return np.vstack(predictions)
    

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
                                                   momentum=self.momentum,
                                                   mu=self.mu,
                                                   eta_plus=self.eta_plus, 
                                                   eta_minus=self.eta_minus,
                                                   decay=self.decay
                                                  )

    # backprop implementation
    def backward(self, error_signals, accumulate=False):
        """Versione modificata per supportare accumulazione"""
        # Se error_signals è uno scalare, convertilo in lista
        if np.isscalar(error_signals):
            error_signals = [error_signals]
        elif isinstance(error_signals, np.ndarray) and error_signals.ndim == 0:
            error_signals = [error_signals.item()]
        elif isinstance(error_signals, np.ndarray) and error_signals.ndim == 1:
            error_signals = list(error_signals)
    
        # Per ogni neurone di output
        for i, neuron in enumerate(self.layers[-1]):
            # Se c'è solo un neurone di output e error_signals ha lunghezza 1
            if i < len(error_signals):
                neuron.compute_delta(error_signals[i])
            else:
                # Questo non dovrebbe succedere se le dimensioni sono corrette
                print(f"WARNING: neurone output {i} ma error_signals ha lunghezza {len(error_signals)}")
                neuron.compute_delta(0.0)  # Default a 0
    
        # Calcola delta per layer nascosti
        for l in range(len(self.layers) - 2, 0, -1):
            for neuron in self.layers[l]:
                neuron.compute_delta(None)
    
        # Aggiorna o accumula gradienti
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
    def fit(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=1, patience=30, min_delta=0.001, verbose=True):

        self.loss_history = {"training": [], "validation": []}
        self.best_val_loss = np.inf  # Reset del miglior validation loss
        self.lr_wait = 0  # Reset del contatore per LR scheduling
    
        # Variabili locali per early stopping
        patience_counter = patience
        local_best_val_loss = np.inf  # Variabile locale per early stopping

        for epoch in range(epochs):
            total_loss = 0.0

            # Mini-batch training
            for start_idx in range(0, len(X_train), batch_size):

                end_idx = min(start_idx + batch_size, len(X_train))
                current_batch_size = end_idx - start_idx

                # Reset accumulators
                self._reset_gradients()

                # Accumulate gradients inside the batch
                for i in range(start_idx, end_idx):
                    xi = X_train[i]
                    yi = y_train[i]

                    y_pred = self.forward(xi)
                    err = self.compute_error_signal(yi, y_pred, loss_type=self.loss_type)
                    self.backward(err, accumulate=True)

                #apply accumulated gradient once
                self._apply_accumulated_gradients(batch_size=current_batch_size)

            # Training loss    
            y_pred_train = self.predict(X_train)
            train_loss_epoch = np.sum(self.compute_loss(y_train, y_pred_train, loss_type=self.loss_type)) / len(X_train)
            self.loss_history["training"].append(train_loss_epoch)

            #Validation loss (non tocca i pesi)
            y_pred_val = self.predict(X_val)
            val_loss = np.sum(self.compute_loss(y_val, y_pred_val, loss_type=self.loss_type))
            avg_val_loss = val_loss / len(y_val)
            self.loss_history["validation"].append(avg_val_loss)

            # Salva i migliori pesi
            if avg_val_loss < local_best_val_loss - min_delta:
                local_best_val_loss = avg_val_loss
                patience_counter = patience  # resetta contatore pazienza

                # Salva i pesi migliori
                for layer in self.layers[1:]:
                    for neuron in layer:
                        neuron.set_best_weights()
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Nuovo miglioramento! Loss: {avg_val_loss:.6f}")
            else:
                patience_counter -= 1

            # Early stopping
            if patience_counter <= 0:
                #ripristina i migliori pesi
                for layer in self.layers[1:]:
                    for neuron in layer:
                        neuron.restore_best_weights()
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                    print(f"  Best validation loss: {local_best_val_loss:.6f}")
                    print(f"  Final validation loss: {avg_val_loss:.6f}")
                return self.loss_history
                

        # Learning rate scheduling (usa l'attributo della classe)
        self._update_lr_on_plateau(avg_val_loss)
        # Soglia minima di miglioramento
        if epoch % 25 == 0 and verbose:
            print(f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {self.eta:.6f}")
    
        return self.loss_history
    
    def _update_lr_on_plateau(self, val_loss):
        """Aggiorna il learning rate se la loss di validation non migliora"""
        # Soglia minima di miglioramento
        improvement_threshold = 0.001
    
        if val_loss < self.best_val_loss - improvement_threshold:
            self.best_val_loss = val_loss
            self.lr_wait = 0
            if self.debug:
                print(f"Miglioramento! New best val loss: {val_loss:.4f}")
        else:
            self.lr_wait += 1
    
        # Riduci LR solo dopo 20 epoche senza miglioramento (non 10)
        if self.lr_wait >= 20:
            old_eta = self.eta
            self.eta *= self.decay
            self.lr_wait = 0
            if self.debug:
                print(f"Learning rate ridotto: {old_eta:.6f} -> {self.eta:.6f}") 

    def compute_loss(self, y_true, y_pred, loss_type="half_mse"):
        """
        Calcola la loss in base al tipo specificato.
        """
        # Clipping per stabilità numerica
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        error = y_pred - y_true
        if loss_type == "binary_crossentropy":
            return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif loss_type == "half_mse":
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
        # Clipping per stabilità numerica
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        error = y_pred - y_true
        
        if loss_type == "binary_crossentropy":
        # Per sigmoid/tanh + binary cross-entropy
            return (y_pred - y_true)
        
        elif loss_type == "half_mse":
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
    """
    # a method to save the losses of the training and test sets as a plot
    def save_plots(self, path):
        plt.plot(self.loss_history["training"], label='Training Loss')
        plt.plot(self.loss_history["validation"], label='Validation Loss')
        plt.legend()
        plt.ylim(0, max(max(self.loss_history["training"]), max(self.loss_history["validation"])) * 1.1)
        plt.savefig(path)

    def draw_network(self, path):
        utils.draw_network(self.layers, path)
    """