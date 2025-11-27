from graphviz import Digraph

def draw_network(layers, weights=None, filename="img/network"):
    dot = Digraph(format="png")
    dot.attr(rankdir='LR', splines='line', ranksep='3')

    # create nodes
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

    for layer in range(1, len(layers), 1):
        for neuron in range(len(layers[layer])):
            for connection in range(len(layers[layer - 1])):
                w = layers[layer][neuron].weights[connection]
                color = "red"
                penwidth = str((w - min_w) / (max_w - min_w) * (2 - 0.1) + 0.1)
                label = f"{w:.3f}"
                dot.edge(f"{layer - 1}_{connection}", f"{layer}_{neuron}", color=color, penwidth=penwidth, taillabel=label, labelangle="0", labeldistance="7")


    dot.render(filename, cleanup=True)