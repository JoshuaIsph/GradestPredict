import random


# --- You will need these helper functions ---
# These are the "rules" of your climbing environment.

import numpy as np  # Still useful for reward/heuristic later
def determine_start_and_target(info):
    """
    Determine the start state (LH, RH, LF, RF) and target hold for a given climb.

    Rules:
      • Hands (LH, RH): occupy the lowest one or two green holds.
        - If only one green hold, both hands start there.
      • Feet (LF, RF): occupy the two lowest orange holds.
        - If fewer than two orange holds exist, duplicate as needed.
      • Target: the single highest purple hold (used for is_goal_state).
    """
    coords = info.get("coordinates", [])

    # --- Separate holds by color ---
    green_holds = [h for h in coords if h["color_name"].lower() == "green"]
    orange_holds = [h for h in coords if h["color_name"].lower() == "orange"]
    purple_holds = [h for h in coords if h["color_name"].lower() == "purple"]

    # --- Determine hand start (green holds) ---
    if not green_holds:
        raise ValueError("No green holds found for this climb — cannot determine start hands.")

    green_sorted = sorted(green_holds, key=lambda h: h["y"])
    if len(green_sorted) >= 2:
        LH = green_sorted[0]["hold_id"]
        RH = green_sorted[1]["hold_id"]
    else:
        # Only one green hold — both hands start there
        LH = RH = green_sorted[0]["hold_id"]

    # --- Determine foot start (lowest orange holds) ---
    if orange_holds:
        orange_sorted = sorted(orange_holds, key=lambda h: h["y"])
        if len(orange_sorted) >= 2:
            LF = orange_sorted[0]["hold_id"]
            RF = orange_sorted[1]["hold_id"]
        else:
            LF = RF = orange_sorted[0]["hold_id"]
    else:
        # No orange holds at all (rare)
        LF = RF = None

    # --- Determine target hold (highest purple) ---
    if not purple_holds:
        raise ValueError("No purple holds found for this climb — cannot determine target.")
    target_hold = max(purple_holds, key=lambda h: h["y"])["hold_id"]

    # --- Construct start state ---
    start_state = (LH, RH, LF, RF)

    return start_state, target_hold

def get_valid_moves(current_state, hold_graph, constraints=None):
    """
    Finds all valid (Action, Next_State) pairs from the current_state.

    This version assumes "reach" is pre-calculated in the hold_graph edges.

    'hold_graph' nodes MUST have data:
    - hold_graph.nodes['H1']['pos'] = (x, y)
    - hold_graph.nodes['H1']['type'] = 'hand' | 'foot' | 'any'
    """

    possible_moves = []
    limbs = {'LH': 0, 'RH': 1, 'LF': 2, 'RF': 3}  # Maps limb name to index

    # --- Start Main Logic ---

    # Iterate through each limb you could *move*
    for limb_name, index in limbs.items():

        current_hold_id = current_state[index]

        # Get the holds occupied by the *other* three limbs
        other_limb_holds = list(current_state)
        other_limb_holds.pop(index)  # Remove the current limb's hold

        # --- 1. Loop Through "Reachable" Holds (from Graph) ---
        # This is the key change! We only check neighbors.
        for target_hold_id in hold_graph.neighbors(current_hold_id):

            # --- 2. Run Constraint Checks ---

            # CHECK A: Is the target hold occupied by *another* limb?
            if target_hold_id in other_limb_holds:
                continue  # Target is occupied, invalid move

            # CHECK B: Is this the right *type* of hold?
            try:
                target_type = hold_graph.nodes[target_hold_id]['hold_type']
            except KeyError:
                print(f"Error: Hold '{target_hold_id}' is missing a 'hold_"
                      f"hold_type' attribute.")
                continue

            is_hand_move = 'H' in limb_name

            if is_hand_move and target_type  in ['hand', 'any']:
                continue  # Trying to move a hand to a foot-only hold
            if not is_hand_move and target_type not in ['foot', 'any']:
                continue  # Trying to move a foot to a hand-only hold

            # --- 3. If All Checks Pass, This is a Valid Move ---

            # Construct the action
            action = (limb_name, target_hold_id)

            # Construct the resulting state
            next_state_list = list(current_state)
            next_state_list[index] = target_hold_id
            next_state = tuple(next_state_list)

            possible_moves.append((action, next_state))

    return possible_moves


def calculate_distance_bias_reward(current_state, action, hold_graph, bias_factor):
    """
    Calculates the reward (desirability) of a specific move based on its inverse distance.
    This rewards shorter, 'easier' moves to bias the sampler.

    Args:
        current_state (tuple): The limb configuration before the move.
        action (tuple): (limb_name, target_hold_id).
        hold_graph (networkx.Graph): Graph with edge weights ('weight').
        bias_factor (float): Controls the strength of the bias (e.g., 2.0).

    Returns:
        float: The inverse distance reward for the move.
    """

    limb_moved, target_hold_id = action
    limbs = {'LH': 0, 'RH': 1, 'LF': 2, 'RF': 3}

    # 1. Find the hold that was released
    limb_index = limbs[limb_moved]
    released_hold_id = current_state[limb_index]

    # 2. Get the distance (weight) of the move from the graph
    try:
        # Distance is stored on the edge between the released_hold and the target_hold.
        distance = hold_graph[released_hold_id][target_hold_id]['weight']
    except KeyError:
        print(f"Warning: Edge between {released_hold_id} and {target_hold_id} not found.")
        return 0.0

    # 3. Calculate the Biased Inverse Distance Reward
    if distance == 0:
        return 1000.0  # Highly favor zero-distance moves (safeguard)

    # Reward favors shorter distances (1 / distance^bias_factor)
    reward = 1.0 / (distance ** bias_factor)

    return reward


def calculate_vertical_progress_reward(current_state, next_state, action, hold_graph):
    """
    Calculates the reward based on the vertical (Y-axis) gain of the climber's state.

    Args:
        current_state (tuple): The limb configuration IDs before the move.
        next_state (tuple): The limb configuration IDs after the move.
        action (tuple): (limb_name, target_hold_id).
        hold_graph (networkx.Graph): Graph used to look up coordinates.

    Returns:
        float: A positive reward for vertical gain, or negative/zero for no gain/loss.
    """

    limb_moved, target_hold_id = action
    limbs = {'LH': 0, 'RH': 1, 'LF': 2, 'RF': 3}
    limb_index = limbs[limb_moved]
    released_hold_id = current_state[limb_index]

    # --- 1. Get Coordinates for the Moving Limb ---

    # You need a helper function (like 'get_coords' from earlier) that maps ID to (x, y)
    def get_coords_from_graph(hold_id):
        # Assuming your graph nodes have 'pos' or 'x'/'y' attributes
        node_data = hold_graph.nodes.get(hold_id, {})
        if 'pos' in node_data:
            return node_data['pos']
        return (node_data.get('x', 0), node_data.get('y', 0))  # Fallback if pos missing

    # Y-coordinate of the starting hold
    _, current_y = get_coords_from_graph(released_hold_id)

    # Y-coordinate of the target hold
    _, next_y = get_coords_from_graph(target_hold_id)

    # --- 2. Calculate Gain ---

    y_gain = next_y - current_y

    # --- 3. Return Scaled Reward ---

    # Reward is proportional to the gain. Scale it down so it doesn't overpower the distance reward.
    # Example scaling: 0.2 units of reward per unit of vertical distance gained.
    vertical_progress_reward = 0.2 * y_gain

    return vertical_progress_reward

def calculate_limb_crossing_penalty(current_hold_coords, penalty_value=-5.0):
    """
    Calculates a penalty if the left hand is placed horizontally to the right
    of the right hand, or the left foot is placed to the right of the right foot.

    Args:
        current_hold_coords (dict): Dictionary mapping limb ('LH', 'RH', 'LF', 'RF')
                                    to its (x, y) coordinates.
        penalty_value (float): The base penalty to apply for each crossing instance.
                               Should be a negative value (e.g., -5.0).

    Returns:
        float: The total penalty (negative value, or 0.0 if no crossing).
    """

    total_penalty = 0.0

    # --- Check Hands (LH vs RH) ---
    lh_x = current_hold_coords['LH'][0]
    rh_x = current_hold_coords['RH'][0]

    # If Left Hand (LH) X-coordinate is greater than Right Hand (RH) X-coordinate, they are crossed.
    if lh_x > rh_x:
        total_penalty += penalty_value
        # Optional: Print statement for debugging the agent's bad decisions
        # print("Debug: Hand crossing penalty applied.")

    # --- Check Feet (LF vs RF) ---
    lf_x = current_hold_coords['LF'][0]
    rf_x = current_hold_coords['RF'][0]

    # If Left Foot (LF) X-coordinate is greater than Right Foot (RF) X-coordinate, they are crossed.
    if lf_x > rf_x:
        total_penalty += penalty_value
        # print("Debug: Foot crossing penalty applied.")

    return total_penalty

def calculate_feet_above_hands_penalty(current_hold_coords, penalty_scale=0.5):
    """
    Calculates a penalty if the highest foot is vertically above the lowest hand.

    Args:
        current_hold_coords (dict): Dictionary mapping limb ('LH', 'RH', 'LF', 'RF')
                                    to its (x, y) coordinates.
        penalty_scale (float): Multiplier for the vertical distance difference.
                               Higher scale means a harsher penalty.

    Returns:
        float: A penalty (negative value, or 0.0 if hands are above feet).
    """

    # 1. Gather Y-coordinates
    hand_y = [current_hold_coords['LH'][1], current_hold_coords['RH'][1]]
    foot_y = [current_hold_coords['LF'][1], current_hold_coords['RF'][1]]

    # 2. Find Extremes
    lowest_hand_y = min(hand_y)
    highest_foot_y = max(foot_y)

    # 3. Calculate Vertical Clearance
    # This value is POSITIVE if feet are higher than hands.
    vertical_clearance = highest_foot_y - lowest_hand_y

    total_penalty = 0.0

    if vertical_clearance > 0:
        # Penalty is proportional to how high the feet are above the hands
        total_penalty = -1.0 * penalty_scale * vertical_clearance
        # print("Debug: Feet above hands penalty applied.")

    return total_penalty


def calculate_reward(current_state, action, next_state, hold_graph, bias_factor=2.0):
    """
    Calculates the total reward for a transition (S, A, R, S') by combining
    the base move desirability with critical climbing heuristics.

    Args:
        current_state (tuple): The limb configuration IDs before the move.
        action (tuple): (limb_name, target_hold_id).
        next_state (tuple): The limb configuration IDs after the move.
        hold_graph (networkx.Graph): Graph used to look up weights and coordinates.
        bias_factor (float): Controls the strength of the distance bias.

    Returns:
        float: The final shaped reward for the move.
    """

    # ----------------------------------------------------------------------
    # 📌 REWARD SCALING PARAMETERS (HYPERPARAMETERS)
    # Adjust these values to shape the agent's behavior
    # ----------------------------------------------------------------------
    VERTICAL_PROGRESS_SCALE = 0.2  # Reward per unit of vertical gain
    LIMB_CROSSING_PENALTY = -5.0  # Flat penalty for crossing hands or feet
    HIGH_FEET_PENALTY_SCALE = -0.5  # Penalty per unit of height feet are above hands

    # ----------------------------------------------------------------------

    # 1. --- PREPARE DATA ---

    # Helper function required to safely map hold IDs to (x, y) coordinates
    # (Must be defined in scope or imported)
    def get_coords_from_graph(hold_id):
        node_data = hold_graph.nodes.get(hold_id, {})
        if 'pos' in node_data: return node_data['pos']
        return (node_data.get('x', 0.0), node_data.get('y', 0.0))

    # Convert the new state (S') into a coordinate dictionary for heuristics
    next_hold_coords = {}
    limbs = ['LH', 'RH', 'LF', 'RF']
    for i, limb in enumerate(limbs):
        hold_id = next_state[i]
        next_hold_coords[limb] = get_coords_from_graph(hold_id)

    # 2. --- CALCULATE COMPONENTS ---

    # R1: Base Reward (Favor short, efficient moves)
    R1_distance_bias = calculate_distance_bias_reward(current_state, action, hold_graph, bias_factor)

    # R2: Vertical Progress
    R2_vertical_progress = VERTICAL_PROGRESS_SCALE * calculate_vertical_progress_reward(current_state, next_state,
                                                                                        action, hold_graph)

    # R3: Limb Crossing Penalty
    # Note: LIMB_CROSSING_PENALTY is passed to the function to apply the flat penalty if triggered
    R3_crossing_penalty = calculate_limb_crossing_penalty(next_hold_coords, penalty_value=LIMB_CROSSING_PENALTY)

    # R4: Feet Above Hands Penalty
    # Note: HIGH_FEET_PENALTY_SCALE is used in the function's internal calculation
    R4_high_feet_penalty = calculate_feet_above_hands_penalty(next_hold_coords, penalty_scale=HIGH_FEET_PENALTY_SCALE)

    # 3. --- COMBINE ---

    total_reward = (R1_distance_bias
                    + R2_vertical_progress
                    + R3_crossing_penalty
                    + R4_high_feet_penalty)

    return total_reward
def is_goal_state(state, target_hold):
    """Check if any hand is on the target hold."""
    lh_on_target = (state[0] == target_hold)
    rh_on_target = (state[1] == target_hold)
    return lh_on_target or rh_on_target


# --- Main Function ---

def generate_biased_transitions(hold_graph, start_state, target_hold,
                                num_episodes=3, bias_factor=2.0, max_steps=10):
    """
    Generate a dataset of (S, A, R, S') transitions using biased sampling.

    Args:
        hold_graph (networkx.Graph): Your graph 'G' of holds, with positions and edge weights.
        start_state (tuple): The starting limb configuration, e.g., ('H1', 'H2', 'F1', 'F2').
        target_hold (str): The ID of the final hold, e.g., 'H_Top'.
        num_episodes (int): Number of full climbs (paths) to generate.
        bias_factor (float): How much to prefer "good" moves.
        max_steps (int): Max moves per climb.
    """

    transition_dataset = []  # This is your new dataset!

    for _ in range(num_episodes):
        current_state = start_state
        steps = 0

        while not is_goal_state(current_state, target_hold) and steps < max_steps:

            # 1. Get all *possible* and *valid* moves from this state
            #    This is the core logic you must build.
            #    It uses your hold_graph to check reach and distance.
            possible_moves = get_valid_moves(current_state, hold_graph, constraints={})

            if not possible_moves:
                break  # Reached a dead end

            # 2. Calculate the "reward" (bias) for each possible move
            actions = [move[0] for move in possible_moves]
            next_states = [move[1] for move in possible_moves]

            raw_rewards = [calculate_reward(current_state, a, s_prime, hold_graph, bias_factor)
                           for a, s_prime in possible_moves]

            # 3. Use your bias logic to pick one
            total_reward = sum(raw_rewards)
            if total_reward == 0:
                # All moves have 0 reward, pick uniformly
                probs = [1.0 / len(raw_rewards)] * len(raw_rewards)
            else:
                probs = [r / total_reward for r in raw_rewards]

            # 4. Sample the action and its resulting state
            chosen_index = random.choices(range(len(possible_moves)), weights=probs, k=1)[0]

            chosen_action = actions[chosen_index]
            chosen_next_state = next_states[chosen_index]
            chosen_reward = raw_rewards[chosen_index]  # This is the reward for the *action*

            # 5. Save the full tuple
            # S = current_state
            # A = chosen_action
            # R = chosen_reward
            # S' = chosen_next_state
            transition_dataset.append((current_state, chosen_action, chosen_reward, chosen_next_state))

            # 6. Update for next loop
            current_state = chosen_next_state
            steps += 1

    return transition_dataset