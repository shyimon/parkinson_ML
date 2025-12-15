import numpy as np
import data_manipulation as data
import neural_network as nn
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
import contextlib
import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Batch

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = data.return_monk3(one_hot=True, dataset_shuffle=True)

    # Normalization
    # X_train_normalized = data.normalize(X_train, 0, 1, X_train.min(axis=0), X_train.max(axis=0))
    # X_test_normalized = data.normalize(X_test, 0, 1, X_train.min(axis=0), X_train.max(axis=0))
    X_train_normalized = X_train
    X_val_normalized = X_val
    X_test_normalized = X_test

    batch_size = [1 << i for i in range(X_train_normalized.shape[0].bit_length())]
    batch_size.append(X_train_normalized.shape[0])

    param_grid = {
        "batch_size": [1, 2, 8, 32, X_train_normalized.shape[0]],
        "algorithm": ["sgd", "rprop", "quickprop"],
        "eta": [0.7, 0.9],
        "network_structure": [[2], [3], [4],
                            [4, 2]],
        "decay": [0.9, 0.8],
        "loss_type": ["half_mse", "mae"],
        "l2_reg": [0.0, 0.000001, 0.00001],
        "activation": ["sigmoid", "tanh"],
        "weight_init": ["xavier", "def"]
    }

    param_combinations = list(itertools.product(
        param_grid["batch_size"],
        param_grid["algorithm"],
        param_grid["eta"],
        param_grid["network_structure"],
        param_grid["decay"],
        param_grid["loss_type"],
        param_grid["l2_reg"],
        param_grid["activation"],
        param_grid["weight_init"],
    ))
    
    open('out.txt', 'w').close()
    grid_args = [
        (X_train_normalized, y_train, X_val_normalized, y_val,
        batch_size, algorithm, eta, hidden_layers,
        decay, loss_type, l2_reg, activation, weight_init)
        for (batch_size, algorithm, eta, hidden_layers,
            decay, loss_type, l2_reg, activation, weight_init)
        in param_combinations
    ]

    print(f"Running {len(grid_args)} grid points in parallel...")

    with tqdm_joblib(tqdm(total=len(grid_args), desc="Grid search")):
        all_results = Parallel(n_jobs=-1)(
            delayed(evaluate_grid_point)(args)
            for args in grid_args
        )

    all_results.sort(key=lambda x: x["median_val_loss"])

    with open("out.txt", "w") as f:
        for i, res in enumerate(all_results):
            f.write(f"\n\n======= RUN {i+1} of {len(all_results)} =======")
            f.write(
                f"\nbs={res['batch_size']}, alg={res['algorithm']}, eta={res['eta']}, "
                f"hidden_layers={res['hidden_layers']}, decay={res['decay']}, "
                f"loss={res['loss_type']}, l2={res['l2_reg']}, "
                f"act={res['activation']}, init={res['weight_init']} "
                f"→ median val loss={res['median_val_loss']:.6f}"
            )
            f.write(f"\nIndividual runs: {res['runs']}")
    
    best = min(all_results, key=lambda x: x["median_val_loss"])
    network_structure = [X_train_normalized.shape[1]]
    network_structure.extend(best["hidden_layers"])
    network_structure.append(1)
    final_net = nn.NeuralNetwork(
    network_structure,
    eta=best["eta"],
    loss_type=best["loss_type"],
    l2_lambda=best["l2_reg"],
    algorithm=best["algorithm"],
    activation_type=best["activation"],
    weight_initializer=best["weight_init"],
    decay=best["decay"],
    momentum=0.9
    )
    print("Training final model with best hyperparameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    final_net.fit(
    X_train_normalized,
    y_train,
    X_val_normalized,
    y_val,
    epochs=2000,
    batch_size=best["batch_size"],
    patience=50,
    verbose=False
    )
    
    # Accuracy for the training set
    if best["activation"] == "sigmoid":
        tresh = 0.5
    elif best["activation"] == "tanh":
        tresh = 0.0
        
    print("\nCalculating accuracy...")
    y_pred = final_net.predict(X_train_normalized)
    y_pred_class = np.where(y_pred >= tresh, 1, 0)

    accuracy = np.mean(y_pred_class == y_train) * 100
    print(f"\nFinal Training Accuracy: {accuracy:.2f}%")

    # Accuracy for the test set
    y_pred_test = final_net.predict(X_test_normalized)
    y_pred_test_class = np.where(y_pred_test >= tresh, 1, 0)
    test_accuracy = np.mean(y_pred_test_class == y_test) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print(f"Details:")
    print(f"Correctly predicted training patterns: {np.sum(y_pred_class == y_train)}/{len(y_train)}")
    print(f"Correctly predicted test patterns: {np.sum(y_pred_test_class == y_test)}/{len(y_test)}")

    final_net.save_plots("img/plot.png")
    final_net.draw_network("img/network")
        

def run_single_experiment(X_train, y_train, X_val, y_val, batch_size, algorithm, eta, hidden_layers, decay, loss_type, l2_reg, activation, weight_init):
    network_structure = [X_train.shape[1]]
    for i in range(len(hidden_layers)):
        network_structure.append(hidden_layers[i])
    network_structure.append(y_val.shape[1])

    net = nn.NeuralNetwork(network_structure,
                           eta=eta,
                           loss_type=loss_type,
                           l2_lambda=l2_reg,
                           algorithm=algorithm,
                           decay=decay,
                           activation_type=activation,
                           weight_initalizer=weight_init)
    
    net.fit(X_train, y_train, X_val, y_val, batch_size=batch_size, patience=10, verbose=False)
    return net.best_val_loss

def evaluate_configuration(X_train, y_train, X_val, y_val, batch_size, algorithm, eta, hidden_layers, decay, loss_type, l2_reg, activation, weight_init, n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(
            X_train, y_train, X_val, y_val,
            batch_size, algorithm, eta,
            hidden_layers, decay, loss_type,
            l2_reg, activation, weight_init,
        )
        for i in range(1)
    )

    return np.median(results), results

def evaluate_grid_point(args):
    (X_train, y_train, X_val, y_val,
     batch_size, algorithm, eta, hidden_layers,
     decay, loss_type, l2_reg, activation, weight_init) = args

    median_loss, runs = evaluate_configuration(
        X_train, y_train, X_val, y_val,
        batch_size=batch_size,
        algorithm=algorithm,
        eta=eta,
        hidden_layers=hidden_layers,
        decay=decay,
        loss_type=loss_type,
        l2_reg=l2_reg,
        activation=activation,
        weight_init=weight_init
    )

    return {
        "batch_size": batch_size,
        "algorithm": algorithm,
        "eta": eta,
        "hidden_layers": hidden_layers,
        "decay": decay,
        "loss_type": loss_type,
        "l2_reg": l2_reg,
        "activation": activation,
        "weight_init": weight_init,
        "median_val_loss": median_loss,
        "runs": runs
    }

if __name__ == "__main__":
    main()