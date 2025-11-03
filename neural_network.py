import perceptron_utilities as perceptron

weights = [0.1, 0.2, 0.15, 0.55, 0.01]
x = [1, 5, 7, 2, 3]
bias = 10
threshold = 1
print(perceptron.neuron(x, weights, bias, threshold))
