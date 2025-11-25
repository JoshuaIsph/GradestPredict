import torch.nn as nn
import torch.nn.functional as Functional


class ActorCriticNeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128, 64]):
        super().__init__()

        self.layers = nn.ModuleList()
        input_size = state_size

        # Create shared layers dynamically
        for h in hidden_sizes:
            self.layers.append(nn.Linear(input_size, h))
            input_size = h

        # Actor head
        self.actor_fc = nn.Linear(input_size, action_size)
        # Critic head
        self.critic_fc = nn.Linear(input_size, 1)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = Functional.relu(layer(x))

        action_probs = Functional.softmax(self.actor_fc(x), dim=-1)
        state_value = self.critic_fc(x)

        return action_probs, state_value

