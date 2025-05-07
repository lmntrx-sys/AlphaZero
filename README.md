![image](https://github.com/user-attachments/assets/ffea1f32-3415-4bb2-bafc-425e0524aa37)



# AlphaZero Algorithm Implementation

This repository contains an implementation of the AlphaZero algorithm, a powerful reinforcement learning technique developed by DeepMind for achieving superhuman performance in perfect information games like Go, Chess, and Shogi.

## Overview

The AlphaZero algorithm combines a deep neural network with Monte Carlo Tree Search (MCTS) in a self-play training loop. This allows it to learn effective game-playing strategies from scratch, without relying on human knowledge.

**Key Components:**

* **Neural Network:** A deep convolutional neural network (CNN) that takes the game state as input and outputs two components:
    * **Policy ($\mathbf{p}$):** A probability distribution over possible moves from the current state.
    * **Value ($v$):** An estimate of the probability of winning from the current state.
* **Monte Carlo Tree Search (MCTS):** A search algorithm that uses the neural network's policy and value predictions to guide its exploration of the game tree. MCTS simulates many game rollouts, iteratively refining its understanding of the optimal moves.
* **Self-Play:** The algorithm trains by playing games against itself. The outcomes of these games, along with the MCTS search probabilities, are used to improve the neural network's policy and value predictions.

## Algorithm Workflow

1.  **Initialization:** The neural network is initialized with random weights.
2.  **Self-Play:**
    * In each game state, MCTS is performed using the current neural network to guide the search.
    * The move with the highest visit count in the MCTS search is selected as the next move.
    * The game continues until a terminal state (win, loss, or draw) is reached.
    * The trajectory of game states and the final outcome are stored.
3.  **Neural Network Training:**
    * The collected game trajectories are used as training data.
    * The neural network's weights are updated to minimize the loss between its predictions (policy and value) and the MCTS-derived target policy and the actual game outcome.
4.  **Iteration:** Steps 2 and 3 are repeated iteratively, gradually improving the neural network's playing strength.

## Implementation Details

This implementation may include the following features (depending on the specific version):

* **Deep Neural Network Architecture:** The model i used is the ResNet model which takes the board as an image ( 3 x 3 ) tensor with no colour coding
* **MCTS Implementation:** The Monte Carlo tree search algo is such a long topic to describe in one page so i have provided links to the research paper
* **Training Loop:** The process of generating self-play data and updating the neural network. The game of tictactoe has 19638 or 3^9 states sor you can adjust the parameters to build a good agent
* **Evaluation:** Methods for evaluating the performance of the trained agent (e.g., playing against previous versions or other baselines).
* **Game Environment:** This agent is buitl to play the game of tictactoe, which has over 19000 game states.

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow
* NumPy

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/lmntrx-sys/AlphaZero.git
    cd alphazero
    ```
2.  Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt  # If a requirements.txt file exists
    # Or, install manually:
    # pip install tensorflow  
    pip install numpy
    ```

### Usage

1.  **Configure the environment:** Modify the configuration files (if any) to specify game parameters, network architecture, training settings, etc.
2.  **Run the training script:**
    ```bash
    training_script.py
    ```
3.  **Evaluate the trained agent: Use the full training script to play the game using the model**
    ```bash
    AlphaZero.ipynb
    ```
    (The specific script names might vary.)


## Contributing

Contributions to this implementation are welcome! 

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure they are well-tested.
4.  Submit a pull request with a clear description of your changes.


## License

MIT License


## Further Reading

* [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (Original AlphaGo Zero paper)
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (Original AlphaZero paper)
* (Link to relevant blog posts, articles, or other resources)
