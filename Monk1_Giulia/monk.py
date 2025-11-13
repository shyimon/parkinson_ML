from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
x = monk_s_problems.data.features 
y = monk_s_problems.data.targets 

# metadata 
print(monk_s_problems.metadata) 
  
# variable information 
print(monk_s_problems.variables) 

print("Dimensione del dataset:")
print(len(x))
#convert x and y (dataFrame) into a Numpy matric
x = x.to_numpy()
y = y.to_numpy().ravel()
print("Dataset")
print(x)
print("Vettore target")
print(y)
# print(x.columns)

number_train = int(len(x)) * 0.5
print("How many training data")
print(number_train)
number_validation = int(len(x) - number_train) * 0.5
number_test = int(len(x) - number_train) * 0.5
print("How many validation data")
print(number_validation)
print("How many test data")
print(number_test) 

 # sum = number_train + number_validation + number_test
# print(sum)

train_set = x[:216]
validation_set = x[216:324]
test_set = x[324:432]
y_train = y[:216]
y_val   = y[216:324]
y_test  = y[324:432]

# Parameters definition and initialitation
w = np.random.uniform(low=-0.1, high=0.1, size=6)
b = 0

# Learning rate
eta = 6

#Perceptron learning algorithm (Rosenblatt)
epochs = 10
for epoch in range(epochs): 
    for i in range(len(train_set)): 
        x_i = train_set[i] #i-esima riga di train_set
        target_i = y_train[i] 
        
        net_i = np.dot(x_i, w) + b 
        output_i = 1 if net_i >= 0 else 0  
        
        error_i = target_i - output_i  
        w = w + eta * error_i * x_i 
        b = b + eta * error_i

def output(net):
    return np.where(net >=0, 1, 0)

net = train_set @ w + b
final_output = output(net)
print("L'output del percettrone è")
print(final_output)

# Comparison between final output vector and target vector
comparison = (final_output == y_train)
error_idx = np.where(final_output != y_train)[0]
print("Uncorrect indexes:", error_idx)
num_correct = np.sum(final_output == y_train)
print("Number of correct indexes", num_correct,)
num_wrong = np.sum(final_output != y_train)
print("Number of uncorrect indexes", num_wrong,)
accuracy = np.mean(final_output == y_train) * 100
print("Accuracy =", accuracy, "%")

# Check if the perceptron learned the logic rule of MONK_1
""""
def logic_rule(x):
    a1 = x[:, 0]   
    a2 = x[:, 1]   
    a5 = x[:, 4]   
    return np.where((a1 == a2) | (a5 == 1), 1, 0) #where confronta el. x el. degli array
y_train = y[:216]
logic_data = logic_rule(x)
net_data = x @ w + b 
output_data = out(net)
logic_rule_accuracy = np-mean/output_data == logic_data) * 100
print("How much does the perceptron coincide with the logical rule:" , logic_rule_accuracy , "%")
"""