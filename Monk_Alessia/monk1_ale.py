from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

print("\n=== FASE 2: PREPROCESSING: TR/TS SETS ===")

y_series = y.iloc[:, 0] #Converte il DataFrame y in una Series

positive_indices = y_series[y_series == 1].index
negative_indices = y_series[y_series == 0].index

train_positive_indices = positive_indices[:62]
train_negative_indices = negative_indices[:62]

train_indices = train_positive_indices.union(train_negative_indices)

X_train = X_renamed.loc[train_indices]
y_train = y_series.loc[train_indices]

X_test = X_renamed
y_test = y_series

print(f"Divisione completata!")
print(f"Training set: {len(X_train)} esempi")
print(f"Test set: {len(X_test)} esempi")

print("\nDistribuzione training set:")
print(f"Classe 0: {(y_train == 0).sum()} esempi")
print(f"Classe 1: {(y_train == 1).sum()} esempi")

print("\nFASE 2 COMPLETATA")

print("\nFASE 3: BASELINE MODELING")

dt_model = DecisionTreeClassifier(
    random_state=42, 
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=None,
    criterion='entropy'
)
dt_model.fit(X_train, y_train)
print("Modello Decision Tree addestrato con successo")
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Set):") #aiuta a capire se il modello ha un bias verso una specifica classe
print(classification_report(y_test, y_test_pred))

#misura quanto ogni feature contribuisce alle decisioni del modello
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance - MONK Problem 1')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show() #rappresentazione grafica della feature importance 

cm = confusion_matrix(y_test, y_test_pred)
print("Matrice di Confusione (Test Set):")
print(cm)

plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=X_train.columns,
          class_names=['False', 'True'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree - MONK Problem 1')
plt.show()

print("\n=== 3.7 VERIFICA REGOLA MONK-1 ===")

def test_monk1_rule(model, feature_names):
    """Test specifico per verificare se il modello ha appreso la regola MONK-1"""
    
    print("Regola: (head_shape = body_shape) OR (jacket_color = red)")
    
    test1 = pd.DataFrame([[1, 1, 1, 1, 2, 1]], columns=feature_names)
    test2 = pd.DataFrame([[1, 2, 1, 1, 1, 1]], columns=feature_names) 
    test3 = pd.DataFrame([[1, 2, 1, 1, 2, 1]], columns=feature_names)
    
    pred1 = model.predict(test1)[0]
    pred2 = model.predict(test2)[0]
    pred3 = model.predict(test3)[0]
    
    print(f"  Test 1 - head_shape = body_shape: Predizione = {pred1} (atteso: 1)")
    print(f"  Test 2 - jacket_color = red: Predizione = {pred2} (atteso: 1)")
    print(f"  Test 3 - nessuna condizione: Predizione = {pred3} (atteso: 0)")
    
    correct_predictions = (pred1 == 1) and (pred2 == 1) and (pred3 == 0)
    print(f"\nModello ha appreso correttamente la regola: {correct_predictions}")

test_monk1_rule(dt_model, X_train.columns)

print("=== FASE 3 COMPLETATA ===")