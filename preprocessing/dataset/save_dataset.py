# preprocessing/dataset/save_dataset.py
import csv

from database.data import build_hold_lookup


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


def convert_transitions_to_dicts(transitions):
    """
    Converts transitions into a flat dictionary structure INCLUDING coordinates.

    Args:
        transitions: List of (S, A, R, S') tuples.
    Returns:
        List[dict]: Flattened dataset with coordinates.
    """

    # Build the lookup table for all hold coordinates
    hold_lookup = build_hold_lookup()

    processed_data = []
    limbs = ['LH', 'RH', 'LF', 'RF']

    for row_data in transitions:

        # Unpack the 4 or 5 elements based on the transition length
        if len(row_data) == 5:
            current_state, action, reward, next_state, climb_name = row_data
        elif len(row_data) == 4:
            # Fallback for old data or if name wasn't injected (optional, but safer)
            current_state, action, reward, next_state = row_data
            climb_name = "UNKNOWN"
        else:
            # Skip malformed row
            continue

        row = {}

        # --- 0. Add Climb Name ---
        row['Climb_Name'] = climb_name  # 💥 NEW FIELD


        # --- 1. Process Current State (S) ---
        for i, limb in enumerate(limbs):
            hold_id = current_state[i]
            x, y = hold_lookup.get(str(hold_id), (None, None))

            row[f'S_{limb}_ID'] = hold_id
            row[f'S_{limb}_x'] = x
            row[f'S_{limb}_y'] = y

        # --- 2. Process Action (A) ---
        limb_moved, target_id = action
        target_x, target_y = hold_lookup.get(str(target_id), (None, None))

        row['A_Limb'] = limb_moved
        row['A_Target_ID'] = target_id
        row['A_Target_x'] = target_x
        row['A_Target_y'] = target_y

        # --- 3. Reward (R) ---
        row['Reward'] = reward

        # --- 4. Process Next State (S') ---
        for i, limb in enumerate(limbs):
            hold_id = next_state[i]
            x, y = hold_lookup.get(str(hold_id), (None, None))

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


