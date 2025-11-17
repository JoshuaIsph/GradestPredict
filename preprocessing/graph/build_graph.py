import networkx as nx
from math import sqrt


def euclidean_distance(coord1, coord2):
    """Calculates the 2D Euclidean distance between two hold coordinates."""
    # Assuming coord is a dict with 'x' and 'y' keys
    return sqrt((coord1['x'] - coord2['x']) ** 2 + (coord1['y'] - coord2['y']) ** 2)


def determine_hold_type(move_attribute):
    """
    Determines the hold type based on the hold's attribute or color.
    - 'orange' → 'foot'
    - everything else → 'any'
    """
    if move_attribute is None:
        return 'any'

    move = move_attribute.lower()

    if 'orange' in move:
        return 'foot'

    return 'any'



def build_climb_graph_with_reachability(climb_data, max_distance=30, skip_colors=None):
    """
    Builds a graph where nodes are holds and edges exist if within max_distance.

    Args:
        climb_data (dict): Dictionary containing the 'coordinates' list.
        max_distance (int/float): The maximum distance (in data units) for an edge to exist.
        skip_colors (list, optional): Colors to exclude from the graph.

    Nodes: hold_id, pos(x, y), type(hand/foot/any), color, move
    Edges: weight (distance)
    """
    skip_colors = skip_colors or []
    G = nx.Graph()

    # Add nodes
    valid_coords = []
    # Note: Accessing climb_data['coordinates'] directly is correct based on your data snippet
    for coord in climb_data.get('coordinates', []):

        if coord.get('color_name') not in skip_colors:
            # The .get('move', None) ensures safety if the key is missing
            hold_type = determine_hold_type(coord.get('color_name'))


            G.add_node(
                coord['hold_id'],
                pos=(coord['x'], coord['y']),
                hold_type=hold_type,
                color=coord.get('color_name', 'default'),
                move=coord.get('move')
            )
            valid_coords.append(coord)

    # Add edges
    for i, c1 in enumerate(valid_coords):
        for j, c2 in enumerate(valid_coords):
            if i != j:
                distance = euclidean_distance(c1, c2)

                if distance <= max_distance:
                    G.add_edge(
                        c1['hold_id'],
                        c2['hold_id'],
                        weight=distance
                    )

    if nx.is_connected(G):
        print("Graph is connected.")
    else:
        print("Graph is NOT connected.")

    return G