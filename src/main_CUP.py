import numpy as np
import data_manipulation as data
import neural_network as nn
import pandas as pd
import matplotlib.pyplot as plt


cup_train_X, cup_train_y, cup_val_X, cup_val_y, cup_test_X, cup_test_y = data.return_CUP()

y_min_original = cup_train_y.min(axis=0)
y_max_original = cup_train_y.max(axis=0)

train_min = cup_train_X.min(axis=0)
train_max = cup_train_X.max(axis=0)

activation_type = "tanh"

if activation_type=="tanh":
    target_min = -1
    target_max = 1
elif activation_type=="sigmoid":
    target_min = 0
    target_max = 1

cup_test_X = data.normalize(cup_test_X, target_min, target_max, train_min, train_max)
cup_test_y = data.normalize(cup_test_y, target_min, target_max, y_min_original, y_max_original)
cup_val_X = data.normalize(cup_val_X, target_min, target_max, train_min, train_max)
cup_val_y = data.normalize(cup_val_y, target_min, target_max, y_min_original, y_max_original)
cup_train_X = data.normalize(cup_train_X, target_min, target_max, train_min, train_max)
cup_train_y = data.normalize(cup_train_y, target_min, target_max, y_min_original, y_max_original)
# np.savetxt("cuptest.csv", cup_test_X, delimiter=",", fmt='%.4f')

network_structure = [cup_test_X.shape[1]]
network_structure.append(8)
network_structure.append(6)
network_structure.append(cup_test_y.shape[1])
eta = 0.5

print("Creating neural network with huber loss...")
net = nn.NeuralNetwork(network_structure, eta=eta, loss_type="huber", l2_lambda=1e-8, algorithm="sgd", activation_type=activation_type, eta_plus=1.2, eta_minus=0.5, mu=1.75, decay=0.90, weight_initialzer="xavier", momentum=0.2)
print("Start training...")
net.fit(cup_train_X, cup_train_y, cup_val_X, cup_val_y, epochs=100, batch_size=8, patience=50)

print("\n" + "="*60)
print("Valutazione finale (scala originale):")
print("="*60)

raw_pred_test = net.predict(cup_test_X)
# Denormalizzo le predizioni
pred_test_denorm = data.denormalize(raw_pred_test, target_min, target_max, y_min_original, y_max_original)
true_test_denorm = data.denormalize(cup_test_y, target_min, target_max, y_min_original, y_max_original)

final_mse = data.MSE(true_test_denorm, pred_test_denorm)
final_mee = data.MEE(true_test_denorm, pred_test_denorm)

print(f"MSE finale (test set): {final_mse:.6f}")
print(f"MEE finale (test set): {final_mee:.6f}")

# Opzionale su tr per vedere se ci sono overfitting
raw_pred_train = net.predict(cup_train_X)
pred_train_denorm = data.denormalize(raw_pred_train, target_min, target_max, y_min_original, y_max_original)
true_train_denorm = data.denormalize(cup_train_y, target_min, target_max, y_min_original, y_max_original)

train_mee = data.MEE(true_train_denorm, pred_train_denorm)
print(f"MEE finale (train set): {train_mee:.6f}")

cup_TS = data.return_CUP_TS()
print(cup_TS)

cup_TS = data.normalize(cup_TS, target_min, target_max, train_min, train_max)

TS_predict = net.predict(cup_TS)

TS_predict = data.denormalize(TS_predict, target_min, target_max, y_min_original, y_max_original)

ids = np.arange(1, TS_predict.shape[0] + 1, dtype=int)

df_out = pd.DataFrame(
    TS_predict,
    columns=[f"out_{i+1}" for i in range(TS_predict.shape[1])]
)

df_out.insert(0, "id", ids)

df_out.to_csv("cup_predictions.csv", index=False, float_format="%.16f")