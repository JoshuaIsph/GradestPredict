# main_pipeline.py
import os

from database.data import get_climbs_with_coordinates
from preprocessing.NeuralNetworks.ActorCriticTraining import main as train_actor_critic
from preprocessing.dataset.save_dataset import save_dataset_csv, convert_transitions_to_dicts
from preprocessing.graph.build_graph import build_climb_graph_with_reachability
from preprocessing.paths.biased_sampling import determine_start_and_target, generate_biased_transitions
from preprocessing.visualize.visualize_graph import visualize_climb_graph

if __name__ == "__main__":
    filename = "../../data/climb_dataset.csv"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"🧹 Removed old dataset file: {filename}. Starting fresh.")
    climbs = {}
    angles = [30,35,40,45,50,55,60]
    all_transitions = []
    for angle in angles:
        climbs.update(get_climbs_with_coordinates(angle=angle,num_climbs=100))


    for name, info in climbs.items():
        print(f"\n🧗 Processing climb: {name}")

        G = build_climb_graph_with_reachability(info)



        start_state, target_hold = determine_start_and_target(info)
        if start_state is None or target_hold is None:
            print("⚠️  Skipping this climb due to missing start state or target hold.")
            continue

        biased_transitions, G = generate_biased_transitions(
            G,
            start_state,
            target_hold,
            num_episodes=100,
            bias_factor=2.0,
            max_steps=20
        )
        dataset_dicts = convert_transitions_to_dicts(biased_transitions, G)
        all_transitions += biased_transitions


        #visualize_climb_graph(G)

        save_dataset_csv(dataset_dicts, filename)

    print("\n🎉 Finished dataset generation!")

    # --- 9. Train Actor-Critic on the generated dataset ---
    print("\n🚀 Starting Actor-Critic training...")
    #train_actor_critic(csv_path=filename)
