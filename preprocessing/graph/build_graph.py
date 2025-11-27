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

    # Assuming 'green', 'cyan', 'purple' are also treated as 'any' (hand or foot)
    # and only 'orange' is explicitly treated as 'foot'
    if 'orange' in move:
        return 'foot'

    return 'any'


# 🛠️ MODIFIED FUNCTION START 🛠️
def build_climb_graph_with_reachability(climb_data, initial_max_distance=30, skip_colors=None):
    """
    Builds a graph, increasing max_distance iteratively until the graph is fully connected.

    Args:
        climb_data (dict): Dictionary containing the 'coordinates' list.
        initial_max_distance (int/float): The starting maximum distance for an edge.
        skip_colors (list, optional): Colors to exclude from the graph.
    """
    skip_colors = skip_colors or []
    current_max_distance = initial_max_distance

    # Pre-filter holds to avoid re-calculating this list in every loop iteration
    valid_coords = [
        coord for coord in climb_data.get('coordinates', [])
        if coord.get('color_name') not in skip_colors
    ]

    if not valid_coords:
        print("No valid holds to build a graph.")
        return nx.Graph()

    # --- Start Iterative Loop ---

    while True:
        G = nx.Graph()

        # 1. Add Nodes (Always the same)
        for coord in valid_coords:
            hold_type = determine_hold_type(coord.get('color_name'))
            G.add_node(
                coord['hold_id'],
                pos=(coord['x'], coord['y']),
                hold_type=hold_type,
                color=coord.get('color_name', 'default'),
                move=coord.get('move')
            )

        # 2. Add Edges based on current_max_distance
        for i, c1 in enumerate(valid_coords):
            for j, c2 in enumerate(valid_coords):
                if i != j:
                    distance = euclidean_distance(c1, c2)

                    if distance <= current_max_distance:
                        G.add_edge(
                            c1['hold_id'],
                            c2['hold_id'],
                            weight=distance
                        )

        # 3. Check Connectivity
        if nx.is_connected(G):
            return G
        else:
            # --- Increment the distance ---
            # Increase by 5 units per iteration (you can adjust this step size)
            current_max_distance += 5

            # Safety break to prevent infinite loops (e.g., if coordinates are bad)
            if current_max_distance > 500:
                print("🚨 Warning: Maximum distance limit reached (500). Returning disconnected graph.")
                return G