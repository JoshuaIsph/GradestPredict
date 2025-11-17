import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
import math

def tsp(G, start_node=None, target_node=None):
    """
    Compute a Traveling Salesman Path (TSP) that starts at `start_node` and ends at `target_node`
    using a 'dummy node' trick.

    The dummy node connects only to start_node and target_node with distance 0,
    and to all others with infinite weight. This forces the solver to make a path
    from start -> ... -> target naturally.

    Returns the cleaned-up path without the dummy node.
    """
    # Clone graph so we don’t modify the original
    G_temp = G.copy()

    if start_node and target_node:
        dummy = "__DUMMY__"

        # Add dummy node
        G_temp.add_node(dummy)

        # Connect dummy with zero-weight edges to start and target
        G_temp.add_edge(dummy, start_node, weight=0)
        G_temp.add_edge(dummy, target_node, weight=0)

        # Connect dummy to all other nodes with infinite weight
        for node in G_temp.nodes():
            if node not in {dummy, start_node, target_node}:
                G_temp.add_edge(dummy, node, weight=math.inf)

        # Run TSP on the modified graph
        tsp_path = traveling_salesman_problem(G_temp, weight="weight", cycle=True)

        # Remove dummy node from the solution
        if dummy in tsp_path:
            dummy_index = tsp_path.index(dummy)

            # Path looks like [..., start, ..., target, dummy, ...]
            # Rotate path so it starts at start_node and ends at target_node
            cleaned_path = []
            for i in range(len(tsp_path)):
                node = tsp_path[(dummy_index + i + 1) % len(tsp_path)]
                if node == dummy:
                    break
                cleaned_path.append(node)

        else:
            cleaned_path = tsp_path  # fallback

    else:
        # No start or target specified — regular TSP
        cleaned_path = traveling_salesman_problem(G, weight="weight", cycle=False)

    print(f"TSP path ({start_node} → {target_node}): {cleaned_path}")
    return cleaned_path
