# main_pipeline.py
from database.data import get_climbs_with_coordinates
from preprocessing.dataset.graph_to_dataset import paths_to_dataset
from preprocessing.dataset.save_dataset import save_dataset_csv
from preprocessing.graph.build_graph import build_climb_graph_with_reachability
from preprocessing.graph.edge_wheigts import add_edge_weights
from preprocessing.paths.biased_sampling import generate_biased_transitions
from preprocessing.paths.tsp_paths import tsp
from preprocessing.visualize.visualize_graph import visualize_climb_graph

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

if __name__ == "__main__":
    climbs = get_climbs_with_coordinates()
    all_dataset = []

    for name, info in climbs.items():
        print(f"Climb: {name}")
        G = build_climb_graph_with_reachability(info)
        print(f"Climb: {info}")
        visualize_climb_graph(G)

        start_state, target_hold = determine_start_and_target(info)
        print("Start state:", start_state)
        print("Target hold:", target_hold)


        transitions = generate_biased_transitions(G, start_state, target_hold, num_episodes=5)
        for transition in transitions:
            print(transition)
        all_dataset.extend(transitions)


