from graphviz import Digraph

def draw_network(layers, weights=None, filename="img/network"):
    dot = Digraph(format="png")
    dot.attr(rankdir='LR', splines='line', ranksep='3')

    # create nodes
    for i, n_nodes in enumerate(layers):
        with dot.subgraph(name=f"cluster_{i}") as c:
            c.attr(label=f"Layer {i}")
            for j in range(n_nodes):
                c.node(f"{i}_{j}", shape="circle", style="filled", fillcolor="lightgray")

    # create edges (with optional weight styling)
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                w = weights[i][j][k] if weights else 0.2
                # Color and thickness scale with weight
                color = "black"
                penwidth = str(0.5 + abs(w) * 3)

                dot.edge(f"{i}_{j}", f"{i+1}_{k}", color=color, penwidth=penwidth)

    dot.render(filename, cleanup=True)

draw_network([6,4,2,1])