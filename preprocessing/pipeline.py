# main_pipeline.py
import os

from database.data import get_climbs_with_coordinates
from preprocessing.dataset.graph_to_dataset import paths_to_dataset
from preprocessing.dataset.save_dataset import save_dataset_csv, convert_transitions_to_dicts
from preprocessing.graph.build_graph import build_climb_graph_with_reachability
from preprocessing.graph.edge_wheigts import add_edge_weights
from preprocessing.paths.biased_sampling import generate_biased_transitions, determine_start_and_target
from preprocessing.paths.tsp_paths import tsp
from preprocessing.visualize.visualize_graph import visualize_climb_graph


if __name__ == "__main__":
    filename = "climb_dataset.csv"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"🧹 Removed old dataset file: {filename}. Starting fresh.")
    climbs = get_climbs_with_coordinates(40,100)
    all_dataset = []

    for name, info in climbs.items():
        # Build the climb graph with reachability
        G = build_climb_graph_with_reachability(info)
        #visualize graph
        #visualize_climb_graph(G)
        # determine start and target
        start_state, target_hold = determine_start_and_target(info)
        #generate biased transitions
        transitions = generate_biased_transitions(G, start_state, target_hold, num_episodes=5)
        #convert to dicts and save
        dict = convert_transitions_to_dicts(transitions,G)
        # save to csv
        save_dataset_csv(dict,filename)



