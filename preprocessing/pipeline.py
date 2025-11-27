# main_pipeline.py
import glob


from preprocessing.NeuralNetworks.ActorCriticTraining import main as train_actor_critic
from preprocessing.dataset.sampling import generate_dataset

if __name__ == "__main__":
    print("\n🚀 Starting dataset generation (batched)...")
    generate_dataset()
    """
    print("\n📊 Starting Actor-Critic training on all batches...")
    for batch_file in glob.glob("../../data/batches/climb_batch_*.csv"):
        print(f"\nTraining on {batch_file}")
        train_actor_critic(csv_path=batch_file)
    """
    print("\n🌟 All done!")
