from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
X = monk_s_problems.data.features
y = monk_s_problems.data.targets 

print("\nFASE 1: DATA UNDERSTANDING")

print("\n=== 1.1 ANALISI STRUTTURALE ===")
print(f"Dataset shape: {X.shape}")
print(f"Numero di features: {len(X.columns)}")
print(f"Numero di esempi totali: {len(X)}")
print(f"Nomi features originali: {list(X.columns)}")

print("\n=== 1.2 ANALISI FEATURE CATEGORICHE ===")
for feature in X.columns:
    valori_unici = X[feature].unique()
    print(f"{feature}: {len(valori_unici)} valori → {list(valori_unici)}")
    
print("\n=== 1.3 ANALISI TARGET ===")
distribuzione = y.iloc[:, 0].value_counts()
print(f"Distribuzione classi:\n{distribuzione}")
print(f"Proporzioni: {distribuzione / len(y)}")

print("\n=== 1.4 TIPI DI DATI E VALORI MANCANTI ===")
print("Tipi di dati delle features:")
print(X.dtypes)
print(f"\nTipo di dato del target: {y.iloc[:, 0].dtype}")
print("\nValori mancanti:")
print("Features:", X.isnull().sum().sum())
print("Target:", y.isnull().sum().sum())

print("\n=== 1.5 MAPPATURA E COMPRENSIONE SEMANTICA ===")
feature_mapping = {
    'a1': 'head_shape',
    'a2': 'body_shape', 
    'a3': 'is_smiling',
    'a4': 'holding',
    'a5': 'jacket_color',  # 1=red, 2=yellow, 3=green, 4=blue
    'a6': 'has_tie'
}

print("Mapping feature UCI -> MONK:")
for code, name in feature_mapping.items():
    print(f"  {code} = {name}")

print(f"\nSignificato valori jacket_color: 1=red, 2=yellow, 3=green, 4=blue")
print(f"Regola MONK-1: (head_shape = body_shape) OR (jacket_color = red)")
print(f"Traduzione tecnica: (a1 = a2) OR (a5 = 1)")

# Rinomina le colonne per chiarezza
X_renamed = X.rename(columns=feature_mapping)
print(f"\nFeatures rinominate: {list(X_renamed.columns)}")

print("\n=== ANTEPRIMA DATI CON NOMI SIGNIFICATIVI ===")
print("Prime 3 righe features:")
print(X_renamed.head(3))
print("\nPrime 3 righe target:")
print(y.head(3))

print("\nFASE 1 COMPLETATA")