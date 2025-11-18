# 🧗 GradestPredict: Bouldering Movement Prediction & Control

![Project Status](https://img.shields.io/badge/Status-Research%20Prototype-blue)
![Domain](https://img.shields.io/badge/Domain-Indoor%20Bouldering-orange)
![Dataset](https://img.shields.io/badge/Data-Kilter%20Board-success)

**GradestPredict** is a machine learning initiative focused on solving the "Beta Break" problem—predicting the optimal sequence of body movements required to ascend an indoor climbing route.

By leveraging the standardized geometry of the **Kilter Board**, this project aims to train an agent that creates a "mental model" of climbing physics. It utilizes a hybrid architecture that combines **Supervised Learning** (for initial movement prediction) and **Reinforcement Learning** (for strategy refinement).

> **The Core Problem:** A climbing wall is not just a graph of nodes; it is a physics puzzle. Finding the path isn't enough—the agent must understand *how* to move the body to make the path possible.

---

## 🏔️ The Kilter Board Approach

Training a climbing AI usually suffers from a lack of consistent data. Every gym wall is different.

**GradestPredict solves this by starting with the Kilter Board**—a standardized, LED-lit climbing wall used globally.
* **Standardized Grid:** Thousands of identical walls exist worldwide, meaning the physics and geometry are constant.
* **Huge Dataset:** We leverage the massive library of existing Kilter problems to generate our initial training data.
* **Transfer Learning:** By mastering movement on this fixed grid first, the model learns the fundamental "language" of climbing before attempting to generalize to novel, unstructured walls.

---

## 🧠 The Hybrid Strategy

This project moves beyond simple graph search (like A*) by implementing a three-stage learning pipeline designed to mimic how humans learn to climb.

### 1. The "Watcher" (Data & Heuristics)
We generate a foundational dataset of successful climbing sequences. Using the Kilter Board's fixed layout, we employ algorithmic solvers (and potential real-world climb data) to map out valid transitions between holds. This creates a "textbook" of valid moves, limb positions, and center-of-mass adjustments.

### 2. The "Predictor" (Supervised Movement Prediction)
Before the agent attempts to "climb" on its own, it is trained to **predict** the expert moves.
* **Behavioral Cloning:** The neural network observes a board configuration (State) and predicts the most likely next limb movement (Action).
* **Result:** A model that understands climbing *patterns* and *flow*. It can look at a route and instinctively "know" the likely beta (sequence), solving the "Cold Start" problem where pure RL agents struggle to make even the first move.

### 3. The "Solver" (Actor-Critic Reinforcement Learning)
The pre-trained predictor is then deployed into a physics simulation to refine its technique.
* **The Actor:** Executes the predicted movements in a live environment.
* **The Critic:** Evaluates the stability and efficiency of those movements.
* **Fine-Tuning:** Through trial and error, the agent learns to deviate from the "textbook" moves when necessary, discovering novel optimizations for balance and energy conservation that the initial dataset didn't contain.

---

## 🚧 Key Challenges

* **High-Dimensional Control:** Climbing isn't just selecting a hold; it requires precise coordination of four limbs and the center of mass.
* **Sequential Dependency:** A mistake in move #3 might not cause a fall until move #8. The system must learn long-horizon planning.
* **The "Beta" Variety:** There is rarely only one way to climb a route. The model must be robust enough to find *a* solution, even if it differs from the training examples.

---

## 🗺️ Project Roadmap

- [ ] **Environment:** Build a simulation environment representing the Kilter Board geometry.
- [ ] **Data Pipeline:** Ingest and process Kilter layouts into graph/state representations.
- [ ] **Supervised Model:** Train the initial movement predictor (The "Watcher").
- [ ] **RL Integration:** Implement the Actor-Critic architecture for dynamic fine-tuning.
- [ ] **Visualization:** create a visualizer to display the agent's predicted "ghost" climber.

---

## 🤝 Context

This project is an exploration of **Robotics Control** and **AI Planning** applied to the domain of sports science. It attempts to bridge the gap between discrete graph algorithms and continuous control policies.
