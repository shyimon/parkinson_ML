import pandas as pd
import numpy as np

# Monk datasets are returned separately and splitted into training, test, parameters and targets
# ho messo dataset_shuffle=false
def return_monk1(dataset_shuffle=False, one_hot=False):
    monk1_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
    monk1_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk1_train = pd.read_csv(monk1_train_url, header=None, names=column_names, sep="\\s+")
    monk1_test = pd.read_csv(monk1_test_url, header=None, names=column_names, sep="\\s+")

    #ho sistemato per assicurarmi che le colonne tra train set e test set siano ben allineate
    if dataset_shuffle:
        monk1_train = monk1_train.sample(frac=1).reset_index(drop=True)
        monk1_test = monk1_test.sample(frac=1).reset_index(drop=True)

    if one_hot:
        monk1_train = pd.get_dummies(monk1_train, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], dtype=int)
        monk1_test = pd.get_dummies(monk1_test, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], dtype=int)

        #allineo le colonne
        all_columns = sorted(set(monk1_train.columns).union(set(monk1_test.columns)))
        for col in all_columns:
            if col not in monk1_train.columns:
                monk1_train[col] = 0
            if col not in monk1_test.columns:
                monk1_test[col] = 0

        # riordino
        monk1_train = monk1_train[all_columns]
        monk1_test = monk1_test[all_columns]

    monk1_train_X = monk1_train.drop(columns=['class', 'id']).to_numpy()
    monk1_train_y = monk1_train['class'].to_numpy().reshape(-1, 1)
    monk1_test_X = monk1_test.drop(columns=['class', 'id']).to_numpy()
    monk1_test_y = monk1_test['class'].to_numpy().reshape(-1, 1)

    idx = len(monk1_train_X)
    split = int(0.5 * idx)
    monk1_val_X = monk1_train_X[split:]
    monk1_train_X = monk1_train_X[:split]
    monk1_val_y = monk1_train_y[split:]
    monk1_train_y = monk1_train_y[:split]
    return monk1_train_X, monk1_train_y, monk1_val_X, monk1_val_y, monk1_test_X, monk1_test_y


def return_monk2(dataset_shuffle=True, one_hot=False):
    monk2_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train'
    monk2_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk2_train = pd.read_csv(monk2_train_url, header=None, names=column_names, sep="\\s+")
    monk2_test = pd.read_csv(monk2_test_url, header=None, names=column_names, sep="\\s+")

    if dataset_shuffle:
        monk2_train = monk2_train.sample(frac=1).reset_index(drop=True)
        monk2_test = monk2_test.sample(frac=1).reset_index(drop=True)

    if one_hot:
        monk2_train = pd.get_dummies(monk2_train, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
        monk2_test = pd.get_dummies(monk2_test, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    
        #allineo le colonne
        all_columns = sorted(set(monk2_train.columns).union(set(monk2_test.columns)))
        for col in all_columns:
            if col not in monk2_train.columns:
                monk2_train[col] = 0
            if col not in monk2_test.columns:
                monk2_test[col] = 0

        # riordino
        monk2_train = monk2_train[all_columns]
        monk2_test = monk2_test[all_columns]

    monk2_train_X = monk2_train.drop(columns=['class', 'id']).to_numpy()
    monk2_train_y = monk2_train['class'].to_numpy().reshape(-1, 1)
    monk2_test_X = monk2_test.drop(columns=['class', 'id']).to_numpy()
    monk2_test_y = monk2_test['class'].to_numpy().reshape(-1, 1)

    idx = len(monk2_train_X)
    split = int(0.5 * idx)
    monk2_val_X = monk2_train_X[split:]
    monk2_train_X = monk2_train_X[:split]
    monk2_val_y = monk2_train_y[split:]
    monk2_train_y = monk2_train_y[:split]
    return monk2_train_X, monk2_train_y, monk2_val_X, monk2_val_y, monk2_test_X, monk2_test_y

def return_monk3(dataset_shuffle=True, one_hot=True, val_split=0.3):
    monk3_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train'
    monk3_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk3_train = pd.read_csv(monk3_train_url, header=None, names=column_names, sep="\\s+")
    monk3_test = pd.read_csv(monk3_test_url, header=None, names=column_names, sep="\\s+")

    if dataset_shuffle:
        monk3_train = monk3_train.sample(frac=1).reset_index(drop=True)
        monk3_test = monk3_test.sample(frac=1).reset_index(drop=True)

    if one_hot:
        monk3_train = pd.get_dummies(monk3_train, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
        monk3_test = pd.get_dummies(monk3_test, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

        #allineo le colonne
        all_columns = sorted(set(monk3_train.columns).union(set(monk3_test.columns)))
        for col in all_columns:
            if col not in monk3_train.columns:
                monk3_train[col] = 0
            if col not in monk3_test.columns:
                monk3_test[col] = 0

        # riordino
        monk3_train = monk3_train[all_columns]
        monk3_test = monk3_test[all_columns]
        
    monk3_train_X = monk3_train.drop(columns=['class', 'id']).to_numpy()
    monk3_train_y = monk3_train['class'].to_numpy().reshape(-1, 1)
    monk3_test_X = monk3_test.drop(columns=['class', 'id']).to_numpy()
    monk3_test_y = monk3_test['class'].to_numpy().reshape(-1, 1)

    idx = len(monk3_train_X)
    split = int((1 - val_split) * idx)
    monk3_val_X = monk3_train_X[split:]
    monk3_train_X = monk3_train_X[:split]
    monk3_val_y = monk3_train_y[split:]
    monk3_train_y = monk3_train_y[:split]
    return monk3_train_X, monk3_train_y, monk3_val_X, monk3_val_y, monk3_test_X, monk3_test_y


def return_CUP(dataset_shuffle=True, train_size=250, validation_size=125, test_size=125):
    cols = ["id"] + [f"in_{i}" for i in range(1, 13)] + [f"t_{i}" for i in range(1, 5)]
    cup = pd.read_csv("../data/ML-CUP25-TR.csv", comment="#", names=cols) # Quando runnavo da main_CUP_cascade.py non trovava il file 
    pd.set_option("display.precision", 3)

    cup_train = cup[0:train_size]
    cup_val = cup[train_size:train_size+validation_size]
    cup_test = cup[train_size+validation_size:train_size+validation_size+test_size]

    if dataset_shuffle:
        cup_train = cup_train.sample(frac=1).reset_index(drop=True)
        cup_val = cup_val.sample(frac=1).reset_index(drop=True)
        cup_test = cup_test.sample(frac=1).reset_index(drop=True)

    cup_train_X = cup_train.drop(columns=['id', 't_1', 't_2', 't_3', 't_4']).to_numpy()
    cup_train_y = cup_train[['t_1', 't_2', 't_3', 't_4']].to_numpy()
    cup_val_X = cup_val.drop(columns=['id', 't_1', 't_2', 't_3', 't_4']).to_numpy()
    cup_val_y = cup_val[['t_1', 't_2', 't_3', 't_4']].to_numpy()
    cup_test_X = cup_test.drop(columns=['id', 't_1', 't_2', 't_3', 't_4']).to_numpy()
    cup_test_y = cup_test[['t_1', 't_2', 't_3', 't_4']].to_numpy()
    
    return cup_train_X, cup_train_y, cup_val_X, cup_val_y, cup_test_X, cup_test_y

def normalize_dataset(X_train, X_val, X_test, min_val=0, max_val=1):
    """Normalizza tutti i dataset usando min/max del training set"""
    # Converte in float se necessario (per one-hot encoding booleano)
    if X_train.dtype == bool or np.issubdtype(X_train.dtype, np.bool_):
        X_train = X_train.astype(float)
        X_val = X_val.astype(float)
        X_test = X_test.astype(float)
    
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    
    # Evita divisione per zero
    diff = x_max - x_min
    diff[diff == 0] = 1
    
    X_train_norm = (X_train - x_min) / diff * (max_val - min_val) + min_val
    X_val_norm = (X_val - x_min) / diff * (max_val - min_val) + min_val
    X_test_norm = (X_test - x_min) / diff * (max_val - min_val) + min_val
    
    return X_train_norm, X_val_norm, X_test_norm