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


    plt.figure(figsize=(12, 10), dpi=300)  # high-resolution figure
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
    # Draw edge weights + latest reward
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0)
        reward = d.get('latest_reward', None)

        if reward is not None:
            edge_labels[(u, v)] = f"{weight:.2f} | R={reward:.2f}"
        else:
            edge_labels[(u, v)] = f"{weight:.2f}"

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color='red'
    )

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

import matplotlib.pyplot as plt
import networkx as nx


def visualize_path(G, transitions, attempt_index=0, pos_scale=1.5):
    if not transitions:
        print("No transitions to visualize.")
        return

    # --- Get original positions and scale ---
    pos_original = nx.get_node_attributes(G, 'pos')
    pos_init = {node: (x * pos_scale, y * pos_scale) for node, (x, y) in pos_original.items()}

    # Use spring layout to spread nodes apart, starting from scaled positions
    pos = nx.spring_layout(G, pos=pos_init, k=50, iterations=200)
    # Remove 'fixed' parameter so nodes can move
    # k=50 makes nodes more spread out; increase for more spacing
    # iterations=200 for layout convergence

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=100, node_color="lightblue",font_size=8)
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Draw the transitions
    for step, (state, action, reward, next_state) in enumerate(transitions):
        limb, target_hold = action
        limbs = ['LH', 'RH', 'LF', 'RF']
        moved_index = limbs.index(limb)
        source_hold = state[moved_index]

        # Arrow from source to target
        plt.annotate(
            "",
            xy=pos[target_hold],
            xytext=pos[source_hold],
            arrowprops=dict(arrowstyle="->", color='red', lw=1)
        )

        # Label at midpoint with step number, limb, reward
        mid_x = ((pos[source_hold][0] + pos[target_hold][0]) / 2)
        mid_y = ((pos[source_hold][1] + pos[target_hold][1]) / 2)
        plt.text(
            mid_x, mid_y, f"{step+1}\n{limb}\nR={reward:.1f}",
            color="darkgreen", fontsize=5,
            ha='center', va='center'
        )

    plt.title(f"Climb Attempt {attempt_index}")
    plt.axis('off')
    plt.show()

