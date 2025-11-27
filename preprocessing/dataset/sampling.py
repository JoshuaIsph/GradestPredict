import os
import multiprocessing
from functools import partial

from database.data import get_climbs_with_coordinates
from preprocessing.dataset.save_dataset import save_dataset_csv, convert_transitions_to_dicts
from preprocessing.graph.build_graph import build_climb_graph_with_reachability
from preprocessing.paths.biased_sampling import determine_start_and_target, generate_biased_transitions


# --- Worker Function for Parallel Execution ---
def process_single_climb(climb_item, episodes_per_climb, bias_factor, max_steps):
    name, info = climb_item
    try:
        G = build_climb_graph_with_reachability(info)
        start_state, target_hold = determine_start_and_target(info)
        transitions, G_final = generate_biased_transitions(
            G, start_state, target_hold,
            num_episodes=episodes_per_climb,
            bias_factor=bias_factor,
            max_steps=max_steps
        )
        transitions_with_name = [(s, a, r, s_prime, name) for s, a, r, s_prime in transitions]
        return True, transitions_with_name, name, G_final
    except (ValueError, TypeError) as e:
        return False, f"Logic Error: {type(e).__name__} - {str(e)}", name, None
    except Exception as e:
        return False, f"Unexpected Error: {type(e).__name__} - {str(e)}", name, None


# --- Main Dataset Generation Function ---
def generate_dataset(
        angles=(30, 40),
        climbs_per_angle=5000,
        episodes_per_climb=50,
        data_dir="../../data",
        out_filename="climb_dataset.csv",
        bias_factor=2.0,
        max_steps=30,
        batch_size=500,
):
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, out_filename)

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🧹 Removed old dataset file: {output_path}")

    total_good_climbs = 0
    total_skipped_climbs = 0
    last_successful_graph = None

    print("📥 Loading and processing climbs batch-by-batch...")
    len_transitions = 0
    for angle in angles:
        loaded_for_angle = 0
        while loaded_for_angle < climbs_per_angle:
            try:
                remaining = climbs_per_angle - loaded_for_angle
                batch_climbs = get_climbs_with_coordinates(
                    angle=angle,
                    num_climbs=min(batch_size, remaining)
                )
                if not batch_climbs:
                    break

                print(f"🧗 Processing batch of {len(batch_climbs)} climbs for angle {angle}...")

                climb_items = list(batch_climbs.items())
                worker_func = partial(
                    process_single_climb,
                    episodes_per_climb=episodes_per_climb,
                    bias_factor=bias_factor,
                    max_steps=max_steps
                )

                with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
                    results = pool.map(worker_func, climb_items)

                batch_transitions = []
                for is_success, transitions_or_error, name, G_final in results:
                    if is_success:
                        batch_transitions.extend(transitions_or_error)
                        total_good_climbs += 1
                        last_successful_graph = G_final
                    else:
                        total_skipped_climbs += 1
                        print(f"Skipping climb: {name}. {transitions_or_error}")

                # Convert batch to dicts and save immediately
                if batch_transitions:
                    len_transitions += len(batch_transitions)
                    dataset_dicts = convert_transitions_to_dicts(batch_transitions)
                    save_dataset_csv(dataset_dicts, output_path)

                loaded_for_angle += len(batch_climbs)

            except Exception as e:
                print(f"⚠️ Error loading/processing batch for angle {angle}: {e}")
                break

    print(f"🎉 Dataset generation complete: {output_path}")
    print(f"Total valid climbs: {total_good_climbs}")
    print(f"Transitions generated: {len_transitions}")
    print(f"Total skipped/invalid climbs: {total_skipped_climbs}")

    return output_path
