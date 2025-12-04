import numpy as np
import neuron as nrn

class CascadeNetwork:
    def __init__(self, n_inputs, n_outputs, learning_rate=0.1):
        """
        Inizializza la rete per la Fase 0 in cui non c'è nessuna unità nascosta.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        
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

    def forward(self, input_pattern):
        outputs = []
        for n in self.output_neurons:
            out = n.feed_neuron(input_pattern)
        outputs.append(out)
        return np.array(outputs)