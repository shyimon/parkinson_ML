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

print(len(x))
#convert x and y (dataFrame) into a Numpy matric
x = x.to_numpy()
y = y.to_numpy()
print(x)
print(y)
# print(x.columns)

number_train = int(len(x)) * 0.5
print(number_train)
number_validation = int(len(x) - number_train) * 0.5
number_test = int(len(x) - number_train) * 0.5
print(number_validation)
print(number_test) 

sum = number_train + number_validation + number_test
print(sum)

train_set = x[:216]
validation_set = x[217:324]
test_set = x[325:432]
print(train_set)
print(validation_set)
print(test_set)

#weights definition (matrix) and initialitation
w_1 = np.random.uniform(0.01, 0.1, (216, 6))
print(w_1)
print(w_2)

# input neuron 
# 


