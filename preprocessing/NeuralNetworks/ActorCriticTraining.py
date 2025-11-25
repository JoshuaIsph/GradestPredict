# ActorCriticTraining.py
from sympy.printing.pytorch import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from database.data import get_all_hold_ids
from preprocessing.NeuralNetworks.ActorCriticNeuralNetwork import ActorCriticNeuralNetwork
from preprocessing.dataset.ClimbingDataset import ClimbingCSVDataset


def main(csv_path="../../../data/climb_dataset.csv"):
    # --- 1. CONFIGURATION ---
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 100

    STATE_SIZE = 8
    HIDDEN_SIZES = [128, 128, 64]

    all_hold_ids = get_all_hold_ids()
    ACTION_SIZE = len(all_hold_ids) * 4

    # --- 2. LOAD DATA ---
    dataset = ClimbingCSVDataset(csv_path, all_hold_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = (ActorCriticNeuralNetwork(
        state_size=STATE_SIZE,
        action_size=dataset.num_actions,
        hidden_sizes=HIDDEN_SIZES
    ).to(device))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # --- 4. TRAINING LOOP ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, action_indices in dataloader:
            states = states.to(device)
            action_indices = action_indices.to(device)

            optimizer.zero_grad()
            logits, _ = model(states)
            loss = loss_fn(logits, action_indices)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    # --- 5. SAVE MODEL ---
    torch.save(model.state_dict(), "apprentice_climber.pth")
    print("✅ Model saved as 'apprentice_climber.pth'")

# Only run if executed as a script
if __name__ == "__main__":
    main()
