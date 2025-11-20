import numpy as np
from neural_network import neural_network
import data_manipulation as dt

nn = neural_network()

li =[]
for i in range(6 * 2):
    li.append(np.random.randn())
li = np.reshape(li, (6, 2))
nn.add_layer(li)

li =[]
for i in range(2 * 2):
    li.append(np.random.randn())
li = np.reshape(li, (2, 2))
nn.add_layer(li)

li =[]
for i in range(2 * 1):
    li.append(np.random.randn())
li = np.reshape(li, (2, 1))
nn.add_layer(li)

monk1_train, monk1_test = dt.return_monk1()
nn.train(monk1_train[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']], monk1_train['class'], 1)