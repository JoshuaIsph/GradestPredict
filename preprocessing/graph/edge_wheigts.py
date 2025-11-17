# graph/edge_weights.py

from math import sqrt

def add_edge_weights(G):
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        G.edges[u, v]['weight'] = sqrt((x2 - x1)**2 + (y2 - y1)**2)
