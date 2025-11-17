import random


# --- You will need these helper functions ---
# These are the "rules" of your climbing environment.

import numpy as np  # Still useful for reward/heuristic later


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


import numpy as np


def calculate_reward(current_state, action, next_state, hold_graph, bias_factor):
    """
    Calculates the reward (desirability) of a specific move.

    Reward is based on the inverse of the edge weight (distance)
    to bias sampling towards shorter, 'easier' moves.

    Args:
        current_state (tuple): The limb configuration before the move.
        action (tuple): (limb_name, target_hold_id).
        next_state (tuple): The limb configuration after the move.
        hold_graph (networkx.Graph): Graph with edge weights ('weight').
        bias_factor (float): Controls the strength of the bias (e.g., 2.0).

    Returns:
        float: The desirability/reward for the move.
    """

    limb_moved, target_hold_id = action

    # 1. Find the hold that was released

    # We need to find the ID of the hold the moving limb was previously on.
    # The 'index' corresponds to the limb in the state tuple (0=LH, 1=RH, etc.)
    limbs = {'LH': 0, 'RH': 1, 'LF': 2, 'RF': 3}
    limb_index = limbs[limb_moved]
    released_hold_id = current_state[limb_index]

    # 2. Get the distance (weight) of the move from the graph
    try:
        # Distance is stored on the edge between the released_hold and the target_hold.
        distance = hold_graph[released_hold_id][target_hold_id]['weight']
    except KeyError:
        # This should not happen if get_valid_moves is working correctly,
        # but is a good safeguard.
        print(f"Warning: Edge between {released_hold_id} and {target_hold_id} not found in graph.")
        return 0.0  # Return no reward

    # 3. Calculate the Reward (Biased Inverse Distance)

    # To favor shorter distances (lower cost), we use the inverse (1 / distance).
    # The bias_factor amplifies the preference for short moves.

    # Ensure distance is not zero to prevent division by zero, though unlikely
    # for physical holds.
    if distance == 0:
        return 1000.0  # Highly favor moves to the same hold (e.g., a re-grab), if they were allowed.

    # The final reward value:
    reward = 1.0 / (distance ** bias_factor)

    return reward


def is_goal_state(state, target_hold):
    """Check if any hand is on the target hold."""
    lh_on_target = (state[0] == target_hold)
    rh_on_target = (state[1] == target_hold)
    return lh_on_target or rh_on_target


# --- Main Function ---

def generate_biased_transitions(hold_graph, start_state, target_hold,
                                num_episodes=1, bias_factor=2.0, max_steps=1):
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