import pandas as pd

# Monk datasets are returned separately and splitted into training, test, parameters and targets
def return_monk1():
    monk1_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
    monk1_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk1_train = pd.read_csv(monk1_train_url, header=None, names=column_names, sep="\\s+")
    monk1_test = pd.read_csv(monk1_test_url, header=None, names=column_names, sep="\\s+")

    monk1_train_X = monk1_train.drop(columns=['class', 'id']).to_numpy()
    monk1_train_y = monk1_train['class'].to_numpy()
    monk1_test_X = monk1_test.drop(columns=['class', 'id']).to_numpy()
    monk1_test_y = monk1_test['class'].to_numpy()

    return monk1_train_X, monk1_train_y, monk1_test_X, monk1_test_y


def return_monk2():
    monk2_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train'
    monk2_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk2_train = pd.read_csv(monk2_train_url, header=None, names=column_names, sep="\\s+")
    monk2_test = pd.read_csv(monk2_test_url, header=None, names=column_names, sep="\\s+")

    monk2_train_X = monk2_train.drop(columns=['class', 'id']).to_numpy()
    monk2_train_y = monk2_train['class'].to_numpy()
    monk2_test_X = monk2_test.drop(columns=['class', 'id']).to_numpy()
    monk2_test_y = monk2_test['class'].to_numpy()

    return monk2_train_X, monk2_train_y, monk2_test_X, monk2_test_y


def return_monk3():
    monk3_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train'
    monk3_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk3_train = pd.read_csv(monk3_train_url, header=None, names=column_names, sep="\\s+")
    monk3_test = pd.read_csv(monk3_test_url, header=None, names=column_names, sep="\\s+")

    monk3_train_X = monk3_train.drop(columns=['class', 'id']).to_numpy()
    monk3_train_y = monk3_train['class'].to_numpy()
    monk3_test_X = monk3_test.drop(columns=['class', 'id']).to_numpy()
    monk3_test_y = monk3_test['class'].to_numpy()

    return monk3_train_X, monk3_train_y, monk3_test_X, monk3_test_y

# Normalization method
def normalize(X, min, max):
    if (X.max(axis=0) - X.min(axis=0)).any == 0:
        raise ZeroDivisionError("Division by zero prevented: the minimum and maximum elements for the training examples are equal, invalid dataset.")
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * (max - min) + min