import numpy as np
import data_manipulation as data
from cascade_correlation import CascadeNetwork
import matplotlib.pyplot as plt
import csv
import datetime

def plot_cup_scatter(y_true, y_pred, title="CUP Predictions vs Targets"):
    """
    Genera uno scatter plot per confrontare target reali e predizioni.
    """
    plt.figure(figsize=(12, 5))
    
    dims = y_true.shape[1]
    for i in range(dims):
        plt.subplot(1, dims, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10, c='blue', label='Predictions')
        
        # Diagonale ideale
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        
        plt.title(f'Output Dimension {i+1}')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show() 
    plt.savefig("cup_scatter.png") 
    
tr_X, tr_y, val_X, val_y, te_X, te_y = data.return_CUP() # Caricamento dati CUP

cup_train_X = np.vstack((tr_X, val_X)) 
cup_train_y = np.vstack((tr_y, val_y)) 

cup_test_X = te_X
cup_test_y = te_y

x_min = cup_train_X.min(axis=0)
x_max = cup_train_X.max(axis=0)

cup_train_X = data.normalize(cup_train_X, -1, 1, x_min, x_max)
cup_test_X = data.normalize(cup_test_X, -1, 1, x_min, x_max)

y_min = cup_train_y.min(axis=0)
y_max = cup_train_y.max(axis=0)

cup_train_y = data.normalize(cup_train_y, -1, 1, y_min, y_max)
cup_test_y = data.normalize(cup_test_y, -1, 1, y_min, y_max)

n_inputs = cup_train_X.shape[1]
n_outputs = cup_train_y.shape[1]

eta = 0.05
patience = 30
max_units = 10

print(f"Creating Cascade Network: {n_inputs} Inputs -> {n_outputs} Outputs")
print("Algorithm: Quickprop")

net = CascadeNetwork(n_inputs, n_outputs, eta, algorithm='quickprop', l2_lambda=0.0)

print("Start training Phase 0 (Linear Training)...")

final_error = net.train(cup_train_X, cup_train_y, X_val=cup_test_X, y_val=cup_test_y, max_epochs=2000, tolerance=0.001, patience=30, max_hidden_units=10)

print(f"Phase 0 ended. Residual Error: {final_error:.5f}")
print("\nCalculating Final Metrics...")

# Ottieni le predizioni della rete (ancora normalizzate tra -1 e 1)
output_test_raw = []
for i in range(len(cup_test_X)):
    output_test_raw.append(net.forward(cup_test_X[i]))
output_test_raw = np.array(output_test_raw)

# Denormalizza Predizioni e Target (per calcolare il MEE reale)
cup_test_y_denorm = data.denormalize(cup_test_y, -1, 1, y_min, y_max)
output_test_denorm = data.denormalize(output_test_raw, -1, 1, y_min, y_max)

cup_train_y_denorm = data.denormalize(cup_train_y, -1, 1, y_min, y_max)
output_train_raw = np.array([net.forward(x) for x in cup_train_X])
output_train_denorm = data.denormalize(output_train_raw, -1, 1, y_min, y_max)

# Calcolo MSE 
mse_test = np.mean(np.sum((cup_test_y - output_test_raw) ** 2, axis=1))
mse_train = np.mean(np.sum((cup_train_y - output_train_raw) ** 2, axis=1))

# Calcolo MEE 
mee_test = net.compute_mee(cup_test_y_denorm, output_test_denorm)
mee_train = net.compute_mee(cup_train_y_denorm, output_train_denorm)

print("-" * 30)
print(f"RISULTATI FINALI CUP:")
print(f"MSE (Normalized) - Train: {mse_train:.5f}, Test: {mse_test:.5f}")
print(f"MEE (Original Scale) - Train: {mee_train:.5f}, Test: {mee_test:.5f}") 
print("-" * 30)

# Grafico Scatter Plot
print("Generazione grafico Scatter Plot...")
plot_cup_scatter(cup_test_y_denorm, output_test_denorm, title="CUP: Test Set Predictions")

def generate_blind_test_predictions(net, blind_file_path, save_path, x_min, x_max, y_min, y_max):
    """
    Genera predizioni corrette per il BLIND TEST SET:
    1. Carica i dati
    2. Normalizza l'input (usando i min/max del training!)
    3. Predice
    4. Denormalizza l'output
    5. Salva nel formato richiesto
    """
    print(f"\n{'='*70}")
    print(f" GENERAZIONE PREDIZIONI BLIND TEST SET")
    print(f"{'='*70}")
    
    try:
        # 1. Carica il file
        raw_data = np.genfromtxt(blind_file_path, delimiter=',', comments='#')
        
        # Separa ID e Features
        blind_ids = raw_data[:, 0]
        blind_X = raw_data[:, 1:]
        
        print(f"Caricati {len(blind_X)} campioni dal blind test.")

        # 2. Normalizza l'input 
        blind_X_norm = data.normalize(blind_X, -1, 1, x_min, x_max)
        
        # 3. Genera predizioni
        blind_pred_raw = []
        for i in range(len(blind_X_norm)):
            blind_pred_raw.append(net.forward(blind_X_norm[i]))
        blind_pred_raw = np.array(blind_pred_raw)
        
        # 4. Denormalizza l'output 
        blind_pred_denorm = data.denormalize(blind_pred_raw, -1, 1, y_min, y_max)
        
        # 5. Salva in formato CSV richiesto
        team_name = "NomeTeamDaInserire" 
        today = datetime.date.today().strftime("%d %b %Y")
        
        with open(save_path, 'w') as f:
            # Header richiesto dalle slide (4 righe)
            f.write("# Alessia Stocco, Cosimo Botticelli, Giulia Tumminello\n")
            f.write(f"# {team_name}\n")
            f.write("# ML-CUP25 v1\n")
            f.write(f"# {today}\n")
            
            # Scrittura dati: ID, out1, out2, out3, out4
            for i in range(len(blind_ids)):
                preds_str = ",".join([f"{val:.6f}" for val in blind_pred_denorm[i]])
                f.write(f"{int(blind_ids[i])},{preds_str}\n")
        
        print(f"\nPredizioni salvate correttamente in: {save_path}")
        
    except Exception as e:
        print(f"Errore nella generazione del blind test: {e}")

# net.save_plots("img/cup_plot.png")
# net.draw_network("img/cup_network")

    generate_blind_test_predictions(
        net=net,
        blind_file_path='data/ML-CUP25-TS.csv',  # Controlla che il percorso sia giusto
        save_path=f'{team_name}_ML-CUP25-TS.csv',
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max
    )