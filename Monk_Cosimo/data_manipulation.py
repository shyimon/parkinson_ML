import pandas as pd

def return_monk1():
    monk1_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
    monk1_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk1_train = pd.read_csv(monk1_train_url, header=None, names=column_names, sep="\\s+")
    monk1_test = pd.read_csv(monk1_test_url, header=None, names=column_names, sep="\\s+")

    return monk1_train, monk1_test


def return_monk2():
    monk2_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train'
    monk2_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk2_train = pd.read_csv(monk2_train_url, header=None, names=column_names, sep="\\s+")
    monk2_test = pd.read_csv(monk2_test_url, header=None, names=column_names, sep="\\s+")

    return monk2_train, monk2_test


def return_monk3():
    monk3_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train'
    monk3_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test'
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    monk3_train = pd.read_csv(monk3_train_url, header=None, names=column_names, sep="\\s+")
    monk3_test = pd.read_csv(monk3_test_url, header=None, names=column_names, sep="\\s+")

    return monk3_train, monk3_test
