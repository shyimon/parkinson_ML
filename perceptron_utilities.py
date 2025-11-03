def __net(x, weights, bias):
    result = []
    for i in range(len(weights)):
        result.append(weights[i] * x[i])
    return sum(result) + bias

def __activation(signal, threshold):
    if signal > threshold:
        return 1
    return 0

def neuron(x, weights, bias, threshold):
    return __activation(__net(x, weights, bias), threshold)