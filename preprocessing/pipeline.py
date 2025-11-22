# main_pipeline.py
import os

from database.data import get_climbs_with_coordinates
from preprocessing.dataset.graph_to_dataset import paths_to_dataset
from preprocessing.dataset.save_dataset import save_dataset_csv, convert_transitions_to_dicts
from preprocessing.graph.build_graph import build_climb_graph_with_reachability

from preprocessing.graph.edge_wheigts import add_edge_weights
from preprocessing.paths.biased_sampling import generate_biased_transitions, determine_start_and_target
from preprocessing.paths.shortes_paths import  astar_climb

from preprocessing.visualize.visualize_graph import visualize_climb_graph


if __name__ == "__main__":
    filename = "climb_dataset.csv"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"🧹 Removed old dataset file: {filename}. Starting fresh.")

    climbs = get_climbs_with_coordinates(40,1)

    for name, info in climbs.items():
        print(f"\n🧗 Processing climb: {name}")

        # -------------------------------
        # 1. Build reachability graph
        # -------------------------------
        G = build_climb_graph_with_reachability(info)

        # -------------------------------
        # 2. Pick start + target
        # -------------------------------
        start_state, target_hold = determine_start_and_target(info)

        # -------------------------------
        # 3. Run A* to get optimal path
        # -------------------------------
        print("🔍 Running A* planning...")
        """
        astar_transitions = astar_plan_states(
            start_state=start_state,
            hold_graph=G,
            target_hold=target_hold,
            avg_hand_step=6.0,
            max_expansions=5000,
            bias_factor=2.0       # if your reward uses it
        )
        """

        #print(f" A* returned {len(astar_transitions)} transitions")

        # ---------------------------------
        # 4. Generate additional biased episodes (optional)
        # ---------------------------------

        biased_transitions, G = generate_biased_transitions(
            G,
            start_state,
            target_hold,
            num_episodes=2,
            bias_factor=2.0,
            max_steps=10
        )


        # ---------------------------------
        # 5. Combine both sources
        # ---------------------------------
        all_transitions = biased_transitions# + astar_transitions

        # ---------------------------------
        # 6. Visualize the climb
        # ---------------------------------
        visualize_climb_graph(G)

        # ---------------------------------
        # 7. Convert transitions to dict format
        # ---------------------------------
        dataset_dicts = convert_transitions_to_dicts(all_transitions, G)

        # ---------------------------------
        # 8. Save (append mode!)
        # ---------------------------------
        save_dataset_csv(dataset_dicts, filename)

    print("\n🎉 Finished dataset generation!")
