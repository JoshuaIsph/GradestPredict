import heapq
from collections import defaultdict

from preprocessing.paths.biased_sampling import get_valid_moves, calculate_reward


# --- A* Search Engine Using Your Reward Function ---
def astar_climb(start_state, target_hold, hold_graph, bias_factor=2.0):
    """
    A* search for climbing using your reward function as cost.

    Args:
        start_state (tuple): initial limb configuration (LH, RH, LF, RF)
        target_hold (int/str): goal hold ID
        hold_graph (networkx.Graph): holds + neighbors
        bias_factor (float): bias for distance reward
    Returns:
        path: list of (action, next_state) tuples
    """

    # Priority queue: (f = g + h, g, state, path)
    open_set = []
    heapq.heappush(open_set, (0, 0, start_state, []))

    # Best known g-value for each state
    g_cost = defaultdict(lambda: float('inf'))
    g_cost[start_state] = 0

    visited = set()

    # --- Heuristic: vertical distance to goal (max hand y → target hold y) ---
    def heuristic(state):
        # Get highest hand Y
        lh_y = hold_graph.nodes[state[0]]['pos'][1]
        rh_y = hold_graph.nodes[state[1]]['pos'][1]
        highest_hand_y = max(lh_y, rh_y)
        target_y = hold_graph.nodes[target_hold]['pos'][1]
        return max(0, target_y - highest_hand_y)

    # --- Goal check ---
    def is_goal(state):
        return state[0] == target_hold or state[1] == target_hold

    while open_set:
        f, g, state, path = heapq.heappop(open_set)

        if state in visited:
            continue
        visited.add(state)

        if is_goal(state):
            return path  # list of (action, next_state)

        # Generate valid moves
        moves = get_valid_moves(state, hold_graph)

        for action, next_state in moves:
            # Cost = negative reward
            cost = -calculate_reward(state, action, next_state, hold_graph, bias_factor)
            new_g = g + cost

            if new_g < g_cost[next_state]:
                g_cost[next_state] = new_g
                new_f = new_g + heuristic(next_state)
                heapq.heappush(open_set, (new_f, new_g, next_state, path + [(action, next_state)]))

    # No path found
    return None
