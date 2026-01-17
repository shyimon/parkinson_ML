import os
from sys import platform
import numpy as np

if platform == "win32":
    graphviz_path = r"C:\Graphviz\bin"
    os.environ["PATH"] = graphviz_path + os.pathsep + os.environ["PATH"]
from graphviz import Digraph

# A method to render and save a graphical representation of the nerwork passed using the graphviz library
def draw_network(layers, path="img/network"):
    dot = Digraph(format="png")
    dot.attr(rankdir='LR', splines='line', ranksep='10')

    for i in range(len(layers)):
        with dot.subgraph(name=f"cluster_{i}") as c:
            if i == 0:
                c.attr(label=f"Input Layer")
            elif i == len(layers) - 1:
                c.attr(label=f"Output Layer")
            else:
                c.attr(label=f"Layer {i}")
            for j in range(len(layers[i])):
                c.node(f"{i}_{j}", shape="circle", style="filled", fillcolor="lightgray")

    max_w = -999
    min_w = 999


    for layer in range(1, len(layers), 1):
        for neuron in range(len(layers[layer])):
            for connection in range(len(layers[layer - 1])):
                if layers[layer][neuron].weights[connection] < min_w:
                    min_w = layers[layer][neuron].weights[connection]
                if layers[layer][neuron].weights[connection] > max_w:
                    max_w = layers[layer][neuron].weights[connection]

    scaled_min = np.log1p(abs(min_w))
    scaled_max = np.log1p(abs(max_w))

    for layer in range(1, len(layers), 1):
        for neuron in range(len(layers[layer])):
            for connection in range(len(layers[layer - 1])):
                w = layers[layer][neuron].weights[connection]
                w_abs = abs(w)
                color = "#DC143C" if w > 0 else "#1E90FF"
                scaled = (w_abs / abs(max_w)) ** 0.4
                penwidth = str(scaled * (2.0 - 0.2) + 0.2)
                label = f"{w:.3f}"
                dot.edge(f"{layer - 1}_{connection}", f"{layer}_{neuron}", color=color, penwidth=penwidth, taillabel=label, labelangle="0", labeldistance="8")


    dot.render(path, cleanup=True)