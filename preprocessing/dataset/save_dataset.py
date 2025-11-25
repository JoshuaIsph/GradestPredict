# preprocessing/dataset/save_dataset.py
import csv


def get_coords(graph, hold_id):
    """Safely retrieves (x, y) from a graph node."""

    try:
        node_data = graph.nodes[hold_id]

        # Check if stored as a tuple 'pos': (x, y)
        if 'pos' in node_data:
            return node_data['pos']
        # Fallback (shouldn't happen if graph is built right)
        return (0.0, 0.0)
    except KeyError:
        return (0.0, 0.0)
# --- 1. Conversion Function ---
def convert_transitions_to_dicts(transitions, hold_graph):
    """
    Converts transitions into a flat dictionary structure INCLUDING coordinates.

    Args:
        transitions: List of (S, A, R, S') tuples.
        hold_graph: The NetworkX graph to look up (x, y) coordinates.
    """
    processed_data = []
    limbs = ['LH', 'RH', 'LF', 'RF']

    for current_state, action, reward, next_state in transitions:
        row = {}

        # --- 1. Process Current State (S) ---
        # current_state is ('ID', 'ID', 'ID', 'ID')
        for i, limb in enumerate(limbs):
            hold_id = current_state[i]
            x, y = get_coords(hold_graph, hold_id)

            row[f'S_{limb}_ID'] = hold_id
            row[f'S_{limb}_x'] = x
            row[f'S_{limb}_y'] = y

        # --- 2. Process Action (A) ---
        # action is ('LimbName', 'TargetID')
        limb_moved, target_id = action
        target_x, target_y = get_coords(hold_graph, target_id)

        row['A_Limb'] = limb_moved
        row['A_Target_ID'] = target_id
        row['A_Target_x'] = target_x
        row['A_Target_y'] = target_y

        # --- 3. Reward (R) ---
        row['Reward'] = reward

        # --- 4. Process Next State (S') ---
        for i, limb in enumerate(limbs):
            hold_id = next_state[i]
            x, y = get_coords(hold_graph, hold_id)

            row[f'S_Prime_{limb}_ID'] = hold_id
            row[f'S_Prime_{limb}_x'] = x
            row[f'S_Prime_{limb}_y'] = y

        processed_data.append(row)

    return processed_data

# --- 2. Saving Function (Your Original Code, slightly modified) ---
def save_dataset_csv(dataset_dicts, filename="climb_dataset.csv"):
    """Saves a list of dictionaries to a CSV file."""

    if not dataset_dicts:
        print("Dataset is empty!")
        return

    # Use the keys from the first dictionary as fieldnames
    fieldnames = list(dataset_dicts[0].keys())

    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset_dicts:
                writer.writerow(row)
        print(f"Dataset successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the CSV: {e}")


