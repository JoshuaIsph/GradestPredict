# paths/shortest_paths.py
import networkx as nx

def shortest_path(G, start_node, target_node):
    """
    Compute the shortest path from start to target using Dijkstra.
    """
    path = nx.shortest_path(G, source=start_node, target=target_node, weight='weight')
    return path
