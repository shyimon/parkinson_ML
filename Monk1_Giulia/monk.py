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
validation_set = x[217:324]
test_set = x[325:432]
print("Training set")
print(train_set)
print("Validation set")
print(validation_set)
print("Test set")
print(test_set)

#weights definition (matrix) and initialitation
w = np.random.uniform(low=-0.01, high=0.1, size=6)
print("Pesi inizializzati")
print(w)

#bias and learning rate definition
b = 0
eta = 0.1

# learning algorithm (perceptron learning rule)


# The non linear neuron 
net = train_set @ w + b
print("Vettore net")
print(net)
# print(type(net))
# print(net.shape)

def out(net):
    return np.where(net >= 0, 1, 0)
output = out(net)
print("Vettore output del neurone")
print(output)
      

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
""""