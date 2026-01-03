import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_manipulation import return_CUP, normalize, MSE, MEE, denormalize
import random
import os
from datetime import datetime

# Crea cartelle per salvare i risultati
os.makedirs("results/cup_random_gs", exist_ok=True)
os.makedirs("results/cup_random_gs/plots", exist_ok=True)

# Carica e prepara i dati CUP
print("Loading CUP dataset...")
cup_train_X, cup_train_y, cup_val_X, cup_val_y, cup_test_X, cup_test_y = return_CUP(
    dataset_shuffle=True, 
    train_size=250, 
    validation_size=125, 
    test_size=125
)

# ==================== CONTROLLO NaN ====================
print("\n" + "="*70)
print("CHECKING FOR NaN VALUES")
print("="*70)

def check_nan(data, name):
    has_nan = np.isnan(data).any()
    if has_nan:
        nan_count = np.isnan(data).sum()
        print(f"❌ {name}:  Found {nan_count} NaN values")
        return True
    else:
        print(f"✓ {name}: No NaN values")
        return False

nan_found = False
nan_found |= check_nan(cup_train_X, "Training X")
nan_found |= check_nan(cup_train_y, "Training y")
nan_found |= check_nan(cup_val_X, "Validation X")
nan_found |= check_nan(cup_val_y, "Validation y")
nan_found |= check_nan(cup_test_X, "Test X")
nan_found |= check_nan(cup_test_y, "Test y")

if nan_found:
    print("\n❌ ERROR: NaN values detected in data!")
    print("Attempting to remove NaN rows...")
    
    train_mask = ~(np.isnan(cup_train_X).any(axis=1) | np.isnan(cup_train_y).any(axis=1))
    cup_train_X = cup_train_X[train_mask]
    cup_train_y = cup_train_y[train_mask]
    
    val_mask = ~(np.isnan(cup_val_X).any(axis=1) | np.isnan(cup_val_y).any(axis=1))
    cup_val_X = cup_val_X[val_mask]
    cup_val_y = cup_val_y[val_mask]
    
    test_mask = ~(np.isnan(cup_test_X).any(axis=1) | np.isnan(cup_test_y).any(axis=1))
    cup_test_X = cup_test_X[test_mask]
    cup_test_y = cup_test_y[test_mask]
    
    print(f"\n✓ NaN rows removed")

print("="*70 + "\n")

# Normalizzazione con range più ampio per evitare saturazione
x_min = cup_train_X.min(axis=0)
x_max = cup_train_X.max(axis=0)
y_min = cup_train_y.min(axis=0)
y_max = cup_train_y.max(axis=0)

# Usa range [-0.8, 0.8] invece di [-1, 1] per evitare saturazione di tanh
cup_train_X_norm = normalize(cup_train_X, -0.8, 0.8, x_min, x_max)
cup_val_X_norm = normalize(cup_val_X, -0.8, 0.8, x_min, x_max)
cup_test_X_norm = normalize(cup_test_X, -0.8, 0.8, x_min, x_max)

cup_train_y_norm = normalize(cup_train_y, -0.8, 0.8, y_min, y_max)
cup_val_y_norm = normalize(cup_val_y, -0.8, 0.8, y_min, y_max)
cup_test_y_norm = normalize(cup_test_y, -0.8, 0.8, y_min, y_max)

print(f"✓ Data normalized to [-0.8, 0.8]")
print(f"Training set:      {cup_train_X_norm.shape}")
print(f"Validation set:  {cup_val_X_norm.shape}")
print(f"Test set:    {cup_test_X_norm.shape}\n")

# Spazio iperparametri OTTIMIZZATO
hyperparameter_space = {
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],  # LR più alti
    'weight_initializer': ['xavier'],
    'network_structure': [
        [12, 60, 4],
        [12, 80, 4],
        [12, 100, 4],
        [12, 50, 30, 4],
        [12, 80, 40, 4],
        [12, 100, 50, 4],
        [12, 120, 60, 4],
    ],
    'activation_type': ['tanh'],
    'patience': [100, 150, 200],  # Più patience
    'epochs': [3000, 5000, 7000],  # Più epoche
    'l2_lambda': [0.0, 0.00005, 0.0001],  # Regolarizzazione più leggera
    'batch_size': [16, 32],  # Batch più grandi per stabilità
    'momentum': [0.85, 0.9, 0.95],  # Momentum alto
    'algorithm': ['sgd']
}

n_random_configs = 50

def calculate_accuracy(y_true, y_pred, threshold=5.0):  # Soglia più stretta
    euclidean_distances = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    correct_predictions = np.sum(euclidean_distances < threshold)
    total_predictions = len(y_true)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

print(f"\n{'='*70}")
print(f"Starting OPTIMIZED Random Grid Search with {n_random_configs} configurations")
print(f"Training Loss: MSE | Validation Loss: MEE")
print(f"{'='*70}\n")

results = []
failed_configs = 0

for i in range(n_random_configs):
    config = {
        'learning_rate': random.choice(hyperparameter_space['learning_rate']),
        'weight_initializer': random.choice(hyperparameter_space['weight_initializer']),
        'network_structure': random.choice(hyperparameter_space['network_structure']),
        'activation_type': random.choice(hyperparameter_space['activation_type']),
        'patience': random.choice(hyperparameter_space['patience']),
        'epochs': random.choice(hyperparameter_space['epochs']),
        'l2_lambda': random.choice(hyperparameter_space['l2_lambda']),
        'batch_size': random.choice(hyperparameter_space['batch_size']),
        'momentum': random.choice(hyperparameter_space['momentum']),
        'algorithm': random.choice(hyperparameter_space['algorithm'])
    }
    
    print(f"\n{'-'*70}")
    print(f"Configuration {i+1}/{n_random_configs}")
    print(f"{'-'*70}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'-'*70}\n")
    
    try:
        model = NeuralNetwork(
            config['network_structure'],
            eta=config['learning_rate'],
            loss_type='mse',
            l2_lambda=config['l2_lambda'],
            algorithm=config['algorithm'],
            activation_type=config['activation_type'],
            weight_initializer=config['weight_initializer'],
            momentum=config['momentum']
        )
        
        model.loss_history = {"training": [], "validation": []}
        model.best_val_loss = np.inf
        model.early_stop_wait = 0
        model.lr_wait = 0
        
        start_time = datetime.now()
        
        for epoch in range(config['epochs']):
            # Training con MSE
            epoch_train_loss = 0.0
            for start_idx in range(0, len(cup_train_X_norm), config['batch_size']):
                end_idx = min(start_idx + config['batch_size'], len(cup_train_X_norm))
                current_batch_size = end_idx - start_idx
                
                model._reset_gradients()
                
                for j in range(start_idx, end_idx):
                    xi = cup_train_X_norm[j]
                    yi = cup_train_y_norm[j]
                    
                    y_pred = model.forward(xi)
                    
                    if np.isnan(y_pred).any():
                        raise ValueError(f"NaN in predictions at epoch {epoch}")
                    
                    sample_loss = model.compute_loss(yi, y_pred, loss_type='mse')
                    epoch_train_loss += np.sum(sample_loss)
                    
                    err = model.compute_error_signal(yi, y_pred, loss_type='mse')
                    err = np.asarray(err).flatten()
                    model.backward(err, accumulate=True)
                
                model._apply_accumulated_gradients(batch_size=current_batch_size)
            
            train_loss_epoch = epoch_train_loss / len(cup_train_X_norm)
            model.loss_history["training"].append(train_loss_epoch)
            
            # Validation con MEE - usando il range corretto
            y_pred_val_norm = model.predict(cup_val_X_norm)
            y_pred_val = denormalize(y_pred_val_norm, -0.8, 0.8, y_min, y_max)
            val_loss_mee = MEE(cup_val_y, y_pred_val)
            model.loss_history["validation"].append(val_loss_mee)
            
            # Learning rate decay meno aggressivo
            model._update_lr_on_plateau(val_loss_mee, patience_lr=config['patience'] // 2)
            
            # Early stopping più permissivo
            min_delta = 0.05  # Delta più grande
            if val_loss_mee < model.best_val_loss - min_delta:
                model.best_val_loss = val_loss_mee
                model.early_stop_wait = 0
                for layer in model.layers:
                    for neuron in layer:
                        neuron.set_best_weights()
            else:
                model.early_stop_wait += 1
            
            if model.early_stop_wait >= config['patience']: 
                for layer in model.layers:
                    for neuron in layer: 
                        neuron.restore_best_weights()
                print(f"Early stopping at epoch {epoch} | Best Val MEE: {model.best_val_loss:.6f}")
                break
            
            if epoch % 300 == 0:
                print(f"Epoch {epoch}, Train Loss (MSE): {train_loss_epoch:.6f}, Val MEE: {val_loss_mee:.6f}, LR: {model.eta:.6f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        best_val_loss_mee = model.best_val_loss
        
        # Test - usando il range corretto
        y_pred_test_norm = model.predict(cup_test_X_norm)
        y_pred_test = denormalize(y_pred_test_norm, -0.8, 0.8, y_min, y_max)
        
        test_mee = MEE(cup_test_y, y_pred_test)
        test_mse = MSE(cup_test_y, y_pred_test)
        
        threshold = 5.0
        accuracy = calculate_accuracy(cup_test_y, y_pred_test, threshold=threshold)
        
        print(f"\n{'='*70}")
        print(f"Results for Configuration {i+1}")
        print(f"{'='*70}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Best Validation MEE: {best_val_loss_mee:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test MEE: {test_mee:.6f}")
        print(f"Test Accuracy (within {threshold}): {accuracy:.2f}%")
        print(f"{'='*70}\n")
        
        result = {
            'config_id': i+1,
            'learning_rate': config['learning_rate'],
            'weight_initializer': config['weight_initializer'],
            'network_structure': str(config['network_structure']),
            'activation_type': config['activation_type'],
            'patience': config['patience'],
            'epochs': config['epochs'],
            'l2_lambda': config['l2_lambda'],
            'batch_size': config['batch_size'],
            'momentum':  config['momentum'],
            'algorithm': config['algorithm'],
            'best_val_loss_mee': best_val_loss_mee,
            'test_mse': test_mse,
            'test_mee': test_mee,
            'accuracy': accuracy,
            'training_time': training_time,
            'model':  model,
            'predictions': y_pred_test
        }
        results.append(result)
        
        # Salva plot
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_history['training'], label='Training Loss (MSE)', linewidth=2)
        plt.plot(model.loss_history['validation'], label='Validation MEE', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Config {i+1}: Training (MSE) vs Validation (MEE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/cup_random_gs/plots/loss_config_{i+1}.png", dpi=200)
        plt.close()
        
    except Exception as e:
        failed_configs += 1
        print(f"\n❌ ERROR in configuration {i+1}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

if len(results) == 0:
    print("\n❌ ERROR: ALL CONFIGURATIONS FAILED!")
    exit(1)

# Salva risultati
results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'predictions']} 
                            for r in results])
results_df.to_csv("results/cup_random_gs/all_results.csv", index=False)

print(f"\n{'='*70}")
print(f"Grid Search completed!")
print(f"Successful:  {len(results)}/{n_random_configs}")
print(f"Failed: {failed_configs}/{n_random_configs}")
print(f"{'='*70}\n")

# Best model
best_result = min(results, key=lambda x: x['best_val_loss_mee'])

print(f"\n{'='*70}")
print("BEST HYPERPARAMETERS")
print(f"{'='*70}")
for key in ['config_id', 'network_structure', 'learning_rate', 'activation_type', 
            'weight_initializer', 'batch_size', 'momentum', 'l2_lambda', 'patience', 'epochs']:
    print(f"{key}: {best_result[key]}")
print(f"\nBest Validation MEE: {best_result['best_val_loss_mee']:.6f}")
print(f"Test MEE: {best_result['test_mee']:.6f}")
print(f"Test Accuracy: {best_result['accuracy']:.2f}%")
print(f"{'='*70}\n")

# Re-train best model
print(f"\n{'='*70}")
print("RE-TRAINING BEST MODEL")
print(f"{'='*70}\n")

best_config = {
    'learning_rate': best_result['learning_rate'],
    'weight_initializer': best_result['weight_initializer'],
    'network_structure': eval(best_result['network_structure']),
    'activation_type': best_result['activation_type'],
    'patience': best_result['patience'],
    'epochs': best_result['epochs'],
    'l2_lambda': best_result['l2_lambda'],
    'batch_size': best_result['batch_size'],
    'momentum': best_result['momentum'],
    'algorithm': best_result['algorithm']
}

final_model = NeuralNetwork(
    best_config['network_structure'],
    eta=best_config['learning_rate'],
    loss_type='mse',
    l2_lambda=best_config['l2_lambda'],
    algorithm=best_config['algorithm'],
    activation_type=best_config['activation_type'],
    weight_initializer=best_config['weight_initializer'],
    momentum=best_config['momentum']
)

final_model.loss_history = {"training": [], "validation": []}
final_model.best_val_loss = np.inf
final_model.early_stop_wait = 0
final_model.lr_wait = 0

start_time = datetime.now()

for epoch in range(best_config['epochs']):
    epoch_train_loss = 0.0
    for start_idx in range(0, len(cup_train_X_norm), best_config['batch_size']):
        end_idx = min(start_idx + best_config['batch_size'], len(cup_train_X_norm))
        current_batch_size = end_idx - start_idx
        
        final_model._reset_gradients()
        
        for j in range(start_idx, end_idx):
            xi = cup_train_X_norm[j]
            yi = cup_train_y_norm[j]
            
            y_pred = final_model.forward(xi)
            sample_loss = final_model.compute_loss(yi, y_pred, loss_type='mse')
            epoch_train_loss += np.sum(sample_loss)
            
            err = final_model.compute_error_signal(yi, y_pred, loss_type='mse')
            err = np.asarray(err).flatten()
            final_model.backward(err, accumulate=True)
        
        final_model._apply_accumulated_gradients(batch_size=current_batch_size)
    
    train_loss_epoch = epoch_train_loss / len(cup_train_X_norm)
    final_model.loss_history["training"].append(train_loss_epoch)
    
    y_pred_val_norm = final_model.predict(cup_val_X_norm)
    y_pred_val = denormalize(y_pred_val_norm, -0.8, 0.8, y_min, y_max)
    val_loss_mee = MEE(cup_val_y, y_pred_val)
    final_model.loss_history["validation"].append(val_loss_mee)
    
    final_model._update_lr_on_plateau(val_loss_mee, patience_lr=best_config['patience'] // 2)
    
    min_delta = 0.05
    if val_loss_mee < final_model.best_val_loss - min_delta: 
        final_model.best_val_loss = val_loss_mee
        final_model.early_stop_wait = 0
        for layer in final_model.layers:
            for neuron in layer:
                neuron.set_best_weights()
    else:
        final_model.early_stop_wait += 1
    
    if final_model.early_stop_wait >= best_config['patience']:
        for layer in final_model.layers:
            for neuron in layer: 
                neuron.restore_best_weights()
        print(f"Early stopping at epoch {epoch} | Best Val MEE: {final_model.best_val_loss:.6f}")
        break
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Train Loss (MSE): {train_loss_epoch:.6f}, Val MEE: {val_loss_mee:.6f}")

training_time_final = (datetime.now() - start_time).total_seconds()

# Final evaluation - usando range corretto
y_pred_train_norm = final_model.predict(cup_train_X_norm)
y_pred_train = denormalize(y_pred_train_norm, -0.8, 0.8, y_min, y_max)
final_train_mse = MSE(cup_train_y, y_pred_train)
final_train_mee = MEE(cup_train_y, y_pred_train)
final_train_accuracy = calculate_accuracy(cup_train_y, y_pred_train, threshold=5.0)

y_pred_val_norm = final_model.predict(cup_val_X_norm)
y_pred_val = denormalize(y_pred_val_norm, -0.8, 0.8, y_min, y_max)
final_val_mee = MEE(cup_val_y, y_pred_val)
final_val_mse = MSE(cup_val_y, y_pred_val)
final_val_accuracy = calculate_accuracy(cup_val_y, y_pred_val, threshold=5.0)

y_pred_test_norm = final_model.predict(cup_test_X_norm)
y_pred_test = denormalize(y_pred_test_norm, -0.8, 0.8, y_min, y_max)
final_test_mse = MSE(cup_test_y, y_pred_test)
final_test_mee = MEE(cup_test_y, y_pred_test)
final_test_accuracy = calculate_accuracy(cup_test_y, y_pred_test, threshold=5.0)

print(f"\n{'='*70}")
print("FINAL MODEL EVALUATION")
print(f"{'='*70}")
print(f"Training Time: {training_time_final:.2f}s")
print(f"\nTraining Set:")
print(f"  MSE: {final_train_mse:.6f}, MEE: {final_train_mee:.6f}, Accuracy: {final_train_accuracy:.2f}%")
print(f"\nValidation Set:")
print(f"  MSE: {final_val_mse:.6f}, MEE: {final_val_mee:.6f}, Accuracy: {final_val_accuracy:.2f}%")
print(f"\nTest Set:")
print(f"  MSE:  {final_test_mse:.6f}, MEE: {final_test_mee:.6f}, Accuracy: {final_test_accuracy:.2f}%")
print(f"{'='*70}\n")

# === GRAFICI ===

# 1. Training/Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(final_model.loss_history['training'], label='Training Loss (MSE)', linewidth=2, color='blue')
plt.plot(final_model.loss_history['validation'], label='Validation Loss (MEE)', linewidth=2, color='orange')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Final Model - Training (MSE) vs Validation (MEE) Loss', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cup_random_gs/final_model_loss.png", dpi=300)
plt.close()

# 2. Histogram
plt.figure(figsize=(12, 6))
accuracies = [r['accuracy'] for r in results]
plt.hist(accuracies, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(final_test_accuracy, color='red', linestyle='--', linewidth=2, 
            label=f'Final Model:  {final_test_accuracy:.2f}%')
plt.xlabel('Accuracy (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Distribution of Model Accuracies (threshold={5.0})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("results/cup_random_gs/accuracy_histogram.png", dpi=300)
plt.close()

# 3. Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Final Model Predictions vs Actual\n(Test MEE: {final_test_mee:.4f}, Accuracy: {final_test_accuracy:.2f}%)', 
             fontsize=16, y=0.995)

for idx, ax in enumerate(axes.flat):
    ax.scatter(cup_test_y[: , idx], y_pred_test[:, idx], alpha=0.6, s=50, edgecolors='black')
    
    min_val = min(cup_test_y[:, idx].min(), y_pred_test[:, idx].min())
    max_val = max(cup_test_y[:, idx].max(), y_pred_test[:, idx].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    
    coeffs = np.polyfit(cup_test_y[:, idx], y_pred_test[:, idx], 1)
    poly = np.poly1d(coeffs)
    sorted_actual = np.sort(cup_test_y[:, idx])
    ax.plot(sorted_actual, poly(sorted_actual), 'g-', linewidth=2, alpha=0.7, 
            label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
    
    ax.set_xlabel(f'Actual t_{idx+1}', fontsize=11)
    ax.set_ylabel(f'Predicted t_{idx+1}', fontsize=11)
    ax.set_title(f'Output Dimension {idx+1}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ss_res = np.sum((cup_test_y[:, idx] - y_pred_test[:, idx])**2)
    ss_tot = np.sum((cup_test_y[:, idx] - cup_test_y[: , idx].mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("results/cup_random_gs/predictions_scatter.png", dpi=300)
plt.close()

# 4. MEE Comparison
plt.figure(figsize=(14, 6))
config_ids = [r['config_id'] for r in results]
test_mees = [r['test_mee'] for r in results]
colors = ['red' if r['config_id'] == best_result['config_id'] else 'steelblue' for r in results]

plt.bar(config_ids, test_mees, color=colors, edgecolor='black', alpha=0.7)
plt.axhline(final_test_mee, color='red', linestyle='--', linewidth=2, 
            label=f'Final Model MEE: {final_test_mee:.4f}')
plt.xlabel('Configuration ID', fontsize=12)
plt.ylabel('Test MEE', fontsize=12)
plt.title('Test MEE Comparison Across All Configurations', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("results/cup_random_gs/mee_comparison.png", dpi=300)
plt.close()

print("\n✓ All plots saved in results/cup_random_gs/")
print("\n" + "="*70)
print("OPTIMIZED GRID SEARCH COMPLETE!")
print("="*70)