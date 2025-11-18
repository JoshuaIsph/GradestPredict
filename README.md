# 🧗 GradestPredict: Bouldering Movement Prediction & Control

![Project Status](https://img.shields.io/badge/Status-Research%20Prototype-blue)
![Domain](https://img.shields.io/badge/Domain-Indoor%20Bouldering-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-A2C%20%7C%20Behavioral%20Cloning-red)
![Dataset](https://img.shields.io/badge/Data-Kilter%20Board-success)

**GradestPredict** is a machine learning initiative focused on solving the **"Beta Break"** problem—predicting the optimal, physically sound sequence of body movements required to ascend an indoor climbing route.

By leveraging the standardized geometry of the **Kilter Board**, this project aims to train an agent that creates a **"mental model" of climbing physics** using a hybrid architecture that combines Supervised Learning and Reinforcement Learning. 

---

## 🧭 Table of Contents

1.  [The Core Problem](#the-core-problem)
2.  [The Hybrid Strategy](#the-hybrid-strategy)
    * [1. The "Watcher" (Data & Heuristics)](#1-the-watcher-data--heuristics)
    * [2. The "Predictor" (Supervised Movement Prediction)](#2-the-predictor-supervised-movement-prediction)
    * [3. The "Solver" (Actor-Critic Reinforcement Learning)](#3-the-solver-actor-critic-reinforcement-learning)
3.  [Getting Started](#getting-started)
4.  [Project Roadmap](#project-roadmap)

---

## 1. The Core Problem

A climbing wall is not just a graph of nodes; it is a **physics puzzle**. Finding the shortest path isn't enough—the agent must understand *how* to move the body to make the path possible. Traditional graph search algorithms fail here because they lack continuous control and stability awareness.

We solve this by focusing on the **Kilter Board**, using its standardized geometry and massive existing problem set to learn the fundamental "language" of climbing movement before attempting to generalize.

---

## 2. The Hybrid Strategy

This project employs a robust three-stage learning pipeline designed to mimic how humans transition from imitation to mastery.

### 1. The "Watcher" (Data & Heuristics)

This phase establishes the project's **foundation** by generating a dataset of physically plausible and efficient climbing sequences. Instead of a simple random walk, we employ a custom **Biased Sampler** that explores the Kilter graph.

#### ⚙️ Reward Shaping: The Climbing Heuristics

To ensure the dataset reflects stable climbing strategy, the desirability of a transition $(S, A, R, S')$ is calculated using a sophisticated, multi-component **Reward Shaping Function**. The reward is a sum of positive incentives and steep penalties:

$$\text{Total Reward} = \text{R}_{\text{Base}} + \text{R}_{\text{Vertical}} + \text{P}_{\text{Cross}} + \text{P}_{\text{FeetAboveHands}}$$

| Component | Purpose | Scaling Parameter |
| :--- | :--- | :--- |
| $\mathbf{R}_{\text{Base}}$ | Favors **shorter, low-effort** moves. | `bias_factor` (Exponent of Inverse Distance) |
| $\mathbf{R}_{\text{Vertical}}$ | Rewards **upward progression** of the moved limb. | `VERTICAL_PROGRESS_SCALE` |
| $\mathbf{P}_{\text{Cross}}$ | Heavily penalizes **crossing** hands ($\text{LH}_{\text{x}} > \text{RH}_{\text{x}}$) or feet. | `LIMB_CROSSING_PENALTY` |
| $\mathbf{P}_{\text{FeetAboveHands}}$ | Penalizes **unstable high foot positions** where feet are above the lowest hand. | `HIGH_FEET_PENALTY_SCALE` |

#### 💾 Data Format
The final dataset is saved incrementally (to prevent RAM overload) and includes not just Hold IDs, but the crucial **$(x, y)$ coordinates** for all four limbs in the current state ($S$) and next state ($S'$), ensuring the Predictor learns continuous spatial relationships.

### 2. The "Predictor" (Supervised Movement Prediction)

The Neural Network (the **Actor**) is initially pre-trained via **Behavioral Cloning** on the rich, heuristically-generated dataset from Phase 1.

* **Input:** 8-dimensional state vector (4 limbs $\times$ 2 coordinates).
* **Output:** A discrete classification over all unique, observed move tuples `(Limb, Target Hold)`.
* **Benefit:** This warm-starts the policy, solving the **"Cold Start" problem** and allowing the agent to instinctively "know" the likely beta or flow of movement.

### 3. The "Solver" (Actor-Critic Reinforcement Learning)

The pre-trained Actor is deployed into a physics simulation environment for dynamic fine-tuning using the **Advantage Actor-Critic (A2C)** algorithm.

* **The Actor:** Executes the predicted movements in the live environment.
* **The Critic:** Introduced here, the Critic learns the **Value Function**—it estimates the cumulative future reward for any given configuration, teaching **long-horizon planning**.
* **Learning:** The Actor is updated to maximize moves that result in a high **Advantage** (where the actual outcome is better than the Critic's prediction), pushing the agent to discover novel, optimal strategies not present in the initial dataset.

---

## 3. Getting Started

This section outlines the basic setup required to run the data generation and training pipeline.

### Prerequisites

* Python 3.8+
* PyTorch (for NN training)
* NetworkX (for graph generation)
* Pandas (for dataset handling)
* A local SQLite database containing Kilter Board hole and placement data (e.g., `kilter.db`).

### Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/gradestpredict.git](https://github.com/yourusername/gradestpredict.git)
cd gradestpredict

# Install dependencies
pip install -r requirements.txt
