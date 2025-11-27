import os
import multiprocessing
from functools import partial
import sys  # Useful for debugging worker errors, though not strictly required

# Note: Removed the unused import from sympy.strategies import tryit

from database.data import get_climbs_with_coordinates
from preprocessing.dataset.save_dataset import save_dataset_csv, convert_transitions_to_dicts
from preprocessing.graph.build_graph import build_climb_graph_with_reachability
from preprocessing.paths.biased_sampling import determine_start_and_target, generate_biased_transitions


# --- Worker Function for Parallel Execution ---
# This function must be defined at the top level to be picklable by multiprocessing.
def process_single_climb(climb_item, episodes_per_climb, bias_factor, max_steps):
    """
    Worker function to process one climb: builds graph, finds start/target,
    and generates transitions. Returns a status flag and data.
    """
    name, info = climb_item

    # Use a large try block to catch all errors and propagate them neatly
    try:
        # 1. Build Graph
        G = build_climb_graph_with_reachability(info)

        # 2. Determine Start/Target
        start_state, target_hold = determine_start_and_target(info)

        # 3. Generate Transitions
        # Note: generate_biased_transitions returns (transitions, G)
        transitions, G_final = generate_biased_transitions(
            G,
            start_state,
            target_hold,
            num_episodes=episodes_per_climb,
            bias_factor=bias_factor,
            max_steps=max_steps,
        )


        # Return success tuple (flag, transitions, name, graph)
        return True, transitions, name, G_final

    except (ValueError, TypeError) as e:
        # Specific exceptions related to climb logic (e.g., transition or start/target issue)
        return False, f"Logic Error: {type(e).__name__} - {str(e)}", name, None

    except Exception as e:
        # Catch any other unexpected errors (e.g., graph building, memory)
        return False, f"Unexpected Error: {type(e).__name__} - {str(e)}", name, None


# --- Main Dataset Generation Function ---
def generate_dataset(
        angles=(30, 35, 40, 45, 50, 55, 60),
        climbs_per_angle=200,
        episodes_per_climb=20,
        data_dir="../../data",
        out_filename="climb_dataset.csv",
        bias_factor=2.0,
        max_steps=30,
):
    """
    Generates all transitions using multiprocessing for efficiency.
    """

    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, out_filename)

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🧹 Removed old dataset file: {output_path}")

    print("📥 Loading climbs from database...")
    climbs = {}
    for angle in angles:
        try:
            climbs.update(get_climbs_with_coordinates(angle=angle, num_climbs=climbs_per_angle))
        except Exception as e:
            print(f"⚠️  Error loading climbs for angle {angle}, skipping this angle. ({e})")
            continue

    print(f"✅ Loaded {len(climbs)} climbs for dataset generation.")

    # --- Multiprocessing Setup ---
    all_transitions = []
    total_good_climbs = 0
    total_skipped_climbs = 0
    last_successful_graph = None

    climb_items = list(climbs.items())

    # Use all available cores minus one, or a minimum of 1
    num_processes = max(1, multiprocessing.cpu_count() - 1)

    print(f"🧠 Starting parallel processing on {len(climb_items)} climbs using {num_processes} cores...")

    # 💥 FIX 1: Use functools.partial to create a picklable worker function
    worker_func = partial(
        process_single_climb,
        episodes_per_climb=episodes_per_climb,
        bias_factor=bias_factor,
        max_steps=max_steps
    )

    # 2. Start the multiprocessing Pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the picklable function over the list of climb items
        results = pool.map(worker_func, climb_items)

    # 3. Collect and process results from all workers
    for is_success, transitions_or_error, name, G_final in results:

        if is_success:
            print(f"🧗 Processing climb: {name} - Success")
            all_transitions.extend(transitions_or_error)
            total_good_climbs += 1
            # Keep track of a valid graph for the final saving step
            last_successful_graph = G_final
        else:
            print(f"Skipping climb (transition issue): {name}. {transitions_or_error}")
            total_skipped_climbs += 1

    # --- Final Saving (CRITICAL for efficiency: runs only once) ---

    # Check if we have any transitions and a valid graph
    if not all_transitions:
        print("\n🚫 WARNING: No valid transitions were generated.")
        return output_path

    if not last_successful_graph:
        # Should not happen if all_transitions is not empty, but good safeguard
        print("\n❌ FATAL ERROR: No valid graph was created for dataset conversion.")
        return output_path

    # Convert everything to dicts
    print(f"\n💾 Saving final dataset with {len(all_transitions)} transitions...")

    # We use the graph from the last successful processing run for conversion
    dataset_dicts = convert_transitions_to_dicts(all_transitions, last_successful_graph)
    save_dataset_csv(dataset_dicts, output_path)

    print(f"🎉 Dataset generation complete: {output_path}")
    print(f"Total valid climbs: {total_good_climbs}")
    print(f"Total skipped/invalid climbs: {total_skipped_climbs}")

    return output_path