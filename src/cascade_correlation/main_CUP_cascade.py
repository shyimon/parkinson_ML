import numpy as np
import matplotlib.pyplot as plt
import data_manipulation as data
from cascade_correlation.cascade_correlation import CascadeNetwork
from joblib import Parallel, delayed
import datetime
import os
import time

def plot_learning_curve(net, title="Best model learning curve"):
    plt.figure(figsize=(10, 6))

    plt.plot(net.loss_history, label='Training Loss (MSE)', color='blue', alpha=0.7)
    
    if hasattr(net, 'val_loss_history') and len(net.val_loss_history) > 0:
        plt.plot(net.val_loss_history, label='Validation Loss (MSE)', color='orange', linestyle='--', alpha=0.7)
    
    if hasattr(net, 'test_loss_history') and len(net.test_loss_history) > 0:
            plt.plot(net.test_loss_history, label='Test MSE (Internal)', color='green', linestyle=':', alpha=0.8, linewidth=1.5)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 
    
    save_dir = os.path.join("src", "cascade_correlation")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, "cup_learning_curve_best.png")
    plt.savefig(filename, dpi=150)
    print(f"Grafico salvato in: {filename}")
    plt.close()

def generate_blind_test_predictions(net, blind_file_path, save_path, x_min, x_max, y_min, y_max, team_name="PlusUltra"):
    print(f"\n--- GENERAZIONE BLIND TEST ---")
    try:
        if not os.path.exists(blind_file_path):
            print(f"File {blind_file_path} non trovato")
            return

        raw_data = np.genfromtxt(blind_file_path, delimiter=',', comments='#')
        if raw_data.ndim == 1: raw_data = raw_data.reshape(1, -1)
        blind_ids = raw_data[:, 0]
        blind_X = raw_data[:, 1:]
        
        # Normalizza usando i min/max del training set
        blind_X_norm = data.normalize(blind_X, -1, 1, x_min, x_max)
        
        preds = []
        for x in blind_X_norm:
            preds.append(net.forward(x))
        preds = np.array(preds)
 
        # Denormalizza output
        preds_denorm = data.denormalize(preds, -1, 1, y_min, y_max)
      
        today = datetime.date.today().strftime("%d %b %Y")
        with open(save_path, 'w') as f:
            f.write("# Alessia Stocco, Cosimo Botticelli, Giulia Tumminello\n")
            f.write(f"# {team_name}\n")
            f.write("# ML-CUP25 v1\n")
            f.write(f"# {today}\n")
            for i in range(len(blind_ids)):
                preds_str = ",".join([f"{val:.6f}" for val in preds_denorm[i]])
                f.write(f"{int(blind_ids[i])},{preds_str}\n")
        print(f"Predizioni salvate in: {save_path}")
    except Exception as e:
        print(f"Errore blind test: {e}")

def train_single_model(seed, tr_X, tr_y, val_X, val_y, te_X, te_y, params):
    np.random.seed(seed)
    # Crea e allena rete
    net = CascadeNetwork(tr_X.shape[1], tr_y.shape[1], learning_rate=params['eta'], l2_lambda=0.0001)
    
    net.train(tr_X, tr_y, X_val=val_X, y_val=val_y, X_test=te_X, y_test=te_y,
              max_epochs=params['epochs'], tolerance=params['tolerance'], 
              patience=params['patience'], max_hidden_units=params['max_units'])
    
    # Calcola MSE finale su validation
    preds = np.array([net.forward(x) for x in val_X])
    val_mse = np.mean((val_y - preds)**2)
    return net, val_mse, seed

if __name__ == "__main__":
    tr_X_raw, tr_y_raw, val_X_raw, val_y_raw, te_X_raw, te_y_raw = data.return_CUP()
    
    all_X = np.vstack((tr_X_raw, val_X_raw, te_X_raw))
    all_y = np.vstack((tr_y_raw, val_y_raw, te_y_raw))
    
    x_min, x_max = all_X.min(axis=0), all_X.max(axis=0)
    y_min, y_max = all_y.min(axis=0), all_y.max(axis=0)
    
    dev_X = np.vstack((tr_X_raw, val_X_raw))
    dev_y = np.vstack((tr_y_raw, val_y_raw))
    
    # Normalizzazione [-1, 1] per la tanh
    dev_X_norm = data.normalize(dev_X, -1, 1, x_min, x_max)
    dev_y_norm = data.normalize(dev_y, -1, 1, y_min, y_max)
    
    internal_test_X_norm = data.normalize(te_X_raw, -1, 1, x_min, x_max)
    internal_test_y_norm = data.normalize(te_y_raw, -1, 1, y_min, y_max)
    
    from sklearn.model_selection import train_test_split
    final_tr_X, final_val_X, final_tr_y, final_val_y = train_test_split(
        dev_X_norm, dev_y_norm, test_size=0.15, random_state=42, shuffle=True
    )

    params = {
        'eta': 0.04000000000000001,      
        'patience': 70,
        'max_units': 20,   
        'tolerance': 0.0001,
        'epochs': 2000,
        'algorithm': "quickprop",
        'momentum': 0.0,
        'mu': 1.5
    }
    
    print("Avvio Training Parallelo...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_single_model)(i, final_tr_X, final_tr_y, final_val_X, final_val_y, internal_test_X_norm, internal_test_y_norm, params) 
        for i in range(5)
    )
    
    # Selezione Migliore
    best_net, best_mse, best_seed = min(results, key=lambda x: x[1])
    print(f"\nVINCITORE: Seed {best_seed} con Val MSE {best_mse:.5f}")

    out_train_norm = np.array([best_net.forward(x) for x in final_tr_X])
    out_train_denorm = data.denormalize(out_train_norm, -1, 1, y_min, y_max)
    tr_y_denorm = data.denormalize(final_tr_y, -1, 1, y_min, y_max) # final_tr_y era normalizzato
    mee_train = data.MEE(tr_y_denorm, out_train_denorm)
    
    out_test_norm = np.array([best_net.forward(x) for x in internal_test_X_norm])
    out_test_denorm = data.denormalize(out_test_norm, -1, 1, y_min, y_max)
    mee_test = data.MEE(te_y_raw, out_test_denorm)

    print(f"\nMETRICHE FINALI:")
    print(f"  MEE Train: {mee_train:.4f}")
    print(f"  MEE Test : {mee_test:.4f}")

    plot_learning_curve(best_net)
    
    generate_blind_test_predictions(best_net, 'data/ML-CUP25-TS.csv', 'PlusUltra_ML-CUP25-TS.csv', x_min, x_max, y_min, y_max)