# 🐍 RF Autonomous Snake Agent

## 📌 Project Overview
This repository chronicles the evolution of a Reinforcement Learning (RL) agent trained to master the classic game of Snake. 

The core objective of this project was to solve the "spatial blindness" problem inherent in basic RL implementations. To demonstrate this, the project is split into two phases, comparing a simple baseline model against an advanced visual model:

* **V1 (The Baseline):** A standard linear Deep Q-Network (DQN). It relies strictly on a small 1D array of logical parameters (immediate danger 1-step ahead, food direction). While it learns quickly, it hits a rigid performance plateau because it cannot plan long-term paths or understand the shape of its own tail.
* **V2 (The Hybrid CNN):** A robust Convolutional Neural Network. By feeding the agent a full 3D visual tensor of the entire board combined with logical reflexes, the V2 agent learns true pathfinding capabilities, avoids geometric traps, and achieves significantly higher scores.

---

## 🎮 Gameplay Comparison

The difference in spatial awareness becomes immediately obvious when watching the agents play:

<table>
  <tr>
    <th width="50%">V2: Hybrid CNN Agent (Advanced)</th>
    <th width="50%">V1: Linear DQN Agent (Baseline)</th>
  </tr>
  <tr>
    <td align="center"><img src="visualizations/v2_gameplay.gif" alt="V2 Gameplay" width="90%"/></td>
    <td align="center"><img src="visualizations/v1_gameplay.gif" alt="V1 Gameplay" width="90%"/></td>
  </tr>
  <tr>
    <td><b>Behavior:</b> Demonstrates long-term spatial planning. It proactively slaloms, leaves escape routes open, and understands the shape of its body to avoid getting trapped.</td>
    <td><b>Behavior:</b> Strictly reactive and short-sighted. It chases the food directly and frequently locks itself into geometric dead-ends as it grows longer.</td>
  </tr>
</table>

---

## 🧠 Architecture & Technical Deep Dive (V2 Model)

To shatter the performance ceiling of the baseline model, the state representation and network architecture were completely redesigned using a **Late Fusion** approach.

### 1. Dual State Representation
The V2 model receives two distinct inputs simultaneously:
* **Visual State (3D Tensor):** A 3-channel matrix providing a comprehensive global view of the board:
    * `Channel 1`: Pinpoints the snake's head.
    * `Channel 2`: Maps all physical obstacles (the growing body and walls).
    * `Channel 3`: A spatial heatmap representing the Manhattan distance to the food, acting as a navigational gradient.
* **Logical State (1D Vector):** A 12-parameter vector providing immediate data on absolute dangers (Up, Down, Left, Right), current movement direction, and absolute food location.

### 2. Network Topology (CNN + Late Fusion)
* **Feature Extraction:** The visual input passes through a Deep Convolutional Neural Network (CNN) utilizing `MaxPool` and `Dropout` layers to extract complex geometric patterns without overfitting.
* **Late Fusion:** The compressed visual feature vector is concatenated with the 12 logical parameters just before the final Fully Connected layers. This synchronizes long-term visual pathfinding with immediate logical survival instincts.

### 3. Training Paradigm
* **Algorithm:** Deep Q-Learning optimized with `Adam` and an `MSELoss` function.
* **Target Network:** Implemented a stable Target Network architecture to provide consistent Q-value targets, preventing catastrophic forgetting.
* **Exploration Strategy:** An Epsilon-greedy mechanism that decays over time, transitioning from chaotic board exploration to confident exploitation.

---

## 📊 Results & Comparative Analysis

The training data highlights the clear trade-off between model complexity and the final performance ceiling:

<table>
  <tr>
    <th width="50%">V2 Training (Hybrid CNN)</th>
    <th width="50%">V1 Training (Linear DQN)</th>
  </tr>
  <tr>
    <td align="center"><img src="visualizations/Training_Dashboard_v2.png" alt="V2 Training Graph" width="100%"/></td>
    <td align="center"><img src="visualizations/Training_Dashboard_v1.png" alt="V1 Training Graph" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Slower to learn, higher ceiling.</b> Processing a full visual state requires significantly more training time (~6,000 games). However, it stabilizes at a high moving average (~33.3) with peak scores exceeding 80. The loss converges beautifully.</td>
    <td><b>Fast to learn, low ceiling.</b> The lightweight model learns quickly but hits a strict plateau after just a few hundred games. It caps at a moving average of ~19.7 and a peak score of 50.</td>
  </tr>
</table>

---

## 🛠️ Installation & Usage

### Prerequisites
Make sure your `requirements.txt` includes the following core libraries:
* `torch` (PyTorch)
* `pygame`
* `numpy`
* `matplotlib`
* `ipython`

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/nadav0912/RF-Snake-game.git](https://github.com/nadav0912/RF-Snake-game.git)
   cd RF-Snake-game
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Running the Agent - 
   To start the training process and watch the AI learn in real-time:
   ```bash
   python agent.py
   ```

---

## 📁 Repository Structure
```bash
RF-Snake-game/
│
├── src/                  # Core source code
│   ├── agent.py          # RL agent logic, state parsing, and memory buffer
│   ├── model.py          # Neural Network architectures and Q-Trainer
│   ├── game.py           # Pygame environment, game loop, and reward logic
│   └── helper.py         # Real-time plotting and visualization utilities
│
├── documentation/        # Project report and presentation slides
├── visualizations/       # Saved training graphs and gameplay GIFs
├── model/                # Saved weights (.pth files - excluded from git)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```