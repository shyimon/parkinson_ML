import numpy as np
import matplotlib.pyplot as plt
import data_manipulation as data
from cascade_correlation import CascadeNetwork
from joblib import Parallel, delayed
import datetime
import os
import time

def plot_cup_scatter(y_true, y_pred, title="CUP Predictions vs Targets"):
    """
    Genera uno scatter plot per confrontare target reali e predizioni.
    Salva il grafico come immagine.
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
    save_dir = os.path.join("src", "cascade_correlation")
    filename = os.path.join(save_dir, "cup_scatter_best.png")
    plt.savefig(filename, dpi=150)
    print(f"Grafico Scatter Plot salvato in: {filename}")
    plt.close()

def plot_learning_curve(net, title="Best Model Learning Curve"):
    """
    Plotta la curva di apprendimento (Loss vs Epoche) per il modello.
    Salva il grafico come immagine.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss
    plt.plot(net.loss_history, label='Training Loss (MSE)', color='blue', alpha=0.7)
    
    # Plot Validation Loss (se esiste)
    if hasattr(net, 'val_loss_history') and len(net.val_loss_history) > 0:
        plt.plot(net.val_loss_history, label='Validation Loss (MSE)', color='orange', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Scala logaritmica per vedere meglio i dettagli finali
    
    save_dir = os.path.join("src", "cascade_correlation")
    filename = os.path.join(save_dir, "cup_learning_curve_best.png")
    plt.savefig(filename, dpi=150)
    print(f"Grafico Learning Curve salvato in: {filename}")
    plt.close()

def generate_blind_test_predictions(net, blind_file_path, save_path, x_min, x_max, y_min, y_max, team_name="PlusUltra"):
    """
    Genera predizioni per il BLIND TEST SET.
    """
    print(f"\n{'='*70}")
    print(f" GENERAZIONE PREDIZIONI BLIND TEST SET")
    print(f"{'='*70}")
    
    try:
        if not os.path.exists(blind_file_path):
            print(f"ERRORE: File {blind_file_path} non trovato!")
            return

        # 1. Carica il file
        raw_data = np.genfromtxt(blind_file_path, delimiter=',', comments='#')
        
        # Gestione caso file vuoto o malformato
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        # Separa ID e Features 
        blind_ids = raw_data[:, 0]
        blind_X = raw_data[:, 1:]
        
        print(f"Caricati {len(blind_X)} campioni dal blind test.")

        # 2. Normalizza l'input (usando i min/max del training!)
        blind_X_norm = data.normalize(blind_X, -1, 1, x_min, x_max)
        
        # 3. Genera predizioni
        blind_pred_raw = []
        for i in range(len(blind_X_norm)):
            blind_pred_raw.append(net.forward(blind_X_norm[i]))
        blind_pred_raw = np.array(blind_pred_raw)
        
        # 4. Denormalizza l'output 
        blind_pred_denorm = data.denormalize(blind_pred_raw, -1, 1, y_min, y_max)
        
        # 5. Salva in formato CSV richiesto
        today = datetime.date.today().strftime("%d %b %Y")
        
        with open(save_path, 'w') as f:
            # Header richiesto
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
        import traceback
        traceback.print_exc()

def train_single_model(seed, tr_X, tr_y, val_X, val_y, params):
    """
    Funzione wrapper per addestrare una singola rete con un seed specifico.
    """
    # Imposta il seed per riproducibilità locale al thread
    np.random.seed(seed)
    
    input_dim = tr_X.shape[1]
    output_dim = tr_y.shape[1]
    
    net = CascadeNetwork(
        input_dim, 
        output_dim, 
        learning_rate=params['eta'], 
        algorithm='quickprop', 
        l2_lambda=0.0
    )
    
    # Addestramento
    # Nota: Usiamo val_X come set di validazione per l'early stopping e l'aggiunta di unità
    net.train(
        tr_X, tr_y, 
        X_val=val_X, y_val=val_y, 
        max_epochs=params['epochs'], 
        tolerance=params['tolerance'], 
        patience=params['patience'], 
        max_hidden_units=params['max_units']
    )
    
    # Calcola MSE finale sul validation set (usato per selezionare il modello migliore)
    output_val_raw = np.array([net.forward(x) for x in val_X])
    val_mse = np.mean((val_y - output_val_raw)**2)
    
    return net, val_mse, seed

if __name__ == "__main__":
    # --- CONFIGURAZIONE TEAM ---
    # ATTENZIONE: Modifica questa stringa con il nickname reale del team!
    TEAM_NAME = "PlusUltra" 
    
    # 1. Caricamento e Preparazione Dati
    print("Caricamento dati CUP...")
    tr_X, tr_y, val_X, val_y, te_X, te_y = data.return_CUP()

    # Unione Training + Validation originali per creare il "Development Set" finale
    dev_X = np.vstack((tr_X, val_X)) 
    dev_y = np.vstack((tr_y, val_y)) 

    # Il Test Set interno (te_X) NON deve essere usato per il training/early stopping
    # Lo teniamo da parte SOLO per il calcolo finale del MEE da mettere nel report.
    internal_test_X = te_X
    internal_test_y = te_y

    # Calcolo min/max per normalizzazione (calcolati sul Development Set)
    x_min = dev_X.min(axis=0)
    x_max = dev_X.max(axis=0)
    y_min = dev_y.min(axis=0)
    y_max = dev_y.max(axis=0)

    # Normalizzazione (-1, 1)
    # Normalizziamo il dev set
    dev_X_norm = data.normalize(dev_X, -1, 1, x_min, x_max)
    dev_y_norm = data.normalize(dev_y, -1, 1, y_min, y_max)
    
    # Normalizziamo il test set interno (per valutazione finale)
    internal_test_X_norm = data.normalize(internal_test_X, -1, 1, x_min, x_max)
    internal_test_y_norm = data.normalize(internal_test_y, -1, 1, y_min, y_max)

    # --- SPLIT INTERNO PER EARLY STOPPING ---
    # Per rispettare le regole ed evitare di usare il Test Set per l'early stopping,
    # dividiamo il Dev Set (Tr+Val originali) in (Nuovo Train + Validation per Early Stopping)
    from sklearn.model_selection import train_test_split
    
    # Usiamo un 15% del development set come validazione per guidare la Cascade
    final_tr_X, final_val_stop_X, final_tr_y, final_val_stop_y = train_test_split(
        dev_X_norm, dev_y_norm, test_size=0.15, random_state=42, shuffle=True
    )

    n_inputs = final_tr_X.shape[1]
    n_outputs = final_tr_y.shape[1]

    # 2. Configurazione Parametri Ottimali (da Grid Search)
    params = {
        'eta': 0.01,            # learning_rate
        'patience': 50,
        'max_units': 20,
        'tolerance': 0.001,
        'epochs': 2000
    }
    
    N_TRIALS = 10 # Numero di retraining paralleli

    print(f"\nConfigurazione Rete:")
    print(f"  Input: {n_inputs}, Output: {n_outputs}")
    print(f"  Parametri: {params}")
    print(f"  Retraining Multiplo: {N_TRIALS} esecuzioni in parallelo")
    print(f"  Training Set size: {len(final_tr_X)} (Usato per i pesi)")
    print(f"  Val Set size (Internal): {len(final_val_stop_X)} (Usato per Early Stopping)")

    # 3. Esecuzione Parallela (Retraining)
    print(f"\nAvvio training parallelo...")
    start_time_global = time.time()
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_single_model)(
            seed=i, 
            tr_X=final_tr_X, tr_y=final_tr_y, 
            val_X=final_val_stop_X, val_y=final_val_stop_y, # Usa lo split interno sicuro
            params=params
        ) for i in range(N_TRIALS)
    )
    
    end_time_global = time.time()
    elapsed_total = end_time_global - start_time_global

    # 4. Selezione Miglior Modello
    best_net = None
    best_mse = float('inf')
    best_seed = -1

    print("\nRisultati Retraining (MSE su Internal Validation Split):")
    for net, mse, seed in results:
        print(f"  Trial {seed}: Val MSE = {mse:.5f} (Hidden Units: {len(net.hidden_units)})")
        if mse < best_mse:
            best_mse = mse
            best_net = net
            best_seed = seed

    print(f"\n{'='*40}")
    print(f"VINCITORE: Seed {best_seed} con MSE {best_mse:.5f}")
    print(f"Tempo totale retraining: {elapsed_total:.2f} secondi")
    print(f"{'='*40}")

    # 5. Valutazione Finale su TEST SET INTERNO (MEE Scala Originale)
    # Calcolo predizioni
    output_train_raw = np.array([best_net.forward(x) for x in final_tr_X])
    # Qui usiamo finalmente il Test Set interno per vedere come va (ma non ha influenzato il training)
    output_test_raw = np.array([best_net.forward(x) for x in internal_test_X_norm])

    # Denormalizzazione
    cup_train_y_denorm = data.denormalize(final_tr_y, -1, 1, y_min, y_max)
    output_train_denorm = data.denormalize(output_train_raw, -1, 1, y_min, y_max)

    cup_test_y_denorm = data.denormalize(internal_test_y, -1, 1, y_min, y_max)
    output_test_denorm = data.denormalize(output_test_raw, -1, 1, y_min, y_max)

    # Calcolo MEE
    mee_train = best_net.compute_mee(cup_train_y_denorm, output_train_denorm)
    mee_test = best_net.compute_mee(cup_test_y_denorm, output_test_denorm)

    print(f"\nMETRICHE FINALI (Best Model - Internal Test):")
    print(f"  MEE Train (Original Scale): {mee_train:.5f}")
    print(f"  MEE Test  (Original Scale): {mee_test:.5f}")
    print("  (Riportare questi valori nelle slide come risultati finali)")

    # 6. Generazione Grafici
    print("\nGenerazione grafici...")
    plot_learning_curve(best_net, title=f"Learning Curve (Best Model - Seed {best_seed})")
    plot_cup_scatter(cup_test_y_denorm, output_test_denorm, title=f"CUP Predictions (Best Model - Seed {best_seed})")

    # 7. Generazione File Blind Test (CSV)
    generate_blind_test_predictions(
        net=best_net,
        blind_file_path='data/ML-CUP25-TS.csv',
        save_path=f'{TEAM_NAME}_ML-CUP25-TS.csv',
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        team_name=TEAM_NAME
    )
    
    # 8. Generazione File Abstract (TXT) - RICHIESTO DALLE REGOLE
    print("\nGenerazione Abstract file...")
    abstract_content = f"""Name: Alessia Stocco, Cosimo Botticelli, Giulia Tumminello
Team: {TEAM_NAME}
Dataset: ML-CUP25 v1
Model: Cascade Correlation Network (Constructive Neural Network)
Validation: Hold-out validation on merged TR+VAL set (85/15 split) for early stopping. Model selection performed via Grid Search.
"""
    abstract_path = f'{TEAM_NAME}_abstract.txt'
    with open(abstract_path, "w") as f:
        f.write(abstract_content)
    print(f"File abstract salvato in: {abstract_path}")

    print("\nEsecuzione completata. Controlla i file generati.")