# visualize/visualize_graph.py
import networkx as nx
import matplotlib.pyplot as plt


def visualize_climb_graph(G, paths=None):
    """
    Visualizes the climb graph with optional highlighted paths and hold type labels.

    Args:
        G (networkx.Graph): The climb graph with node attributes:
                            - 'pos': (x, y)
                            - 'color': string (node color)
                            - 'hold_type': string ('hand', 'foot', 'any')
                            and edge attribute:
                            - 'weight': float
        paths (list of lists, optional): List of node sequences (paths) to highlight.
    """

    # Node positions (x, y)
    pos = {node: data['pos'] for node, data in G.nodes(data=True)}

    # Node colors
    colors = [data.get('color', 'lightgray') for _, data in G.nodes(data=True)]

    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=600,
        edge_color='gray',
        font_size=8
    )

    # Draw edge weights
    edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    #  Draw hold_type labels slightly offset from each node

    hold_type_labels = {
        node: data.get('hold_type', 'any') for node, data in G.nodes(data=True)
    }
    # Offset positions so text doesn’t overlap nodes
    label_pos = {n: (x, y + 4) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=hold_type_labels,
        font_color='black',
        font_size=8,
        verticalalignment='bottom'
    )

    # Draw optional paths
    if paths:
        path_colors = ['blue', 'green', 'orange', 'purple', 'cyan']
        for i, path in enumerate(paths):
            path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            color = path_colors[i % len(path_colors)]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                edge_color=color,
                width=3,
                alpha=0.7
            )

    plt.title("Climbing Reachability Graph with Hold Types")
    plt.axis('on')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()
