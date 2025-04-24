import numpy as np
import tensorflow as tf

# Create the Gaming environment
class TicTacToe:
  def __init__(self) -> None:
    self.row_count = 3
    self.column_count = 3
    self.action_space = np.array([i for i in range(self.row_count * self.column_count)])

  # Initialize the previous states for reference
  def get_initial_state(self):
    return np.zeros((self.row_count, self.column_count))

  # Get the next states
  def get_next_state(self, state, action, player):
    row = action // self.column_count
    column = action % self.column_count
    state[row, column] = player
    return state

  def get_valid_moves(self, state):
    return (state.reshape(-1) == 0).astype(np.uint8)

  # Check win
  def check_game_over(self, state, action):
    if action == None:
      return False

    row = action // self.column_count
    column = action % self.column_count
    player = state[row, column]
    return (np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count)

  def get_game_value(self, state, action):
    if np.sum(self.check_game_over(state, action)):
      return 1, True
    if np.sum(self.get_valid_moves(state)) == 0:
      return 0, True
    return 0, False

  def get_opponent(self, player):
    return -player

  def get_encoded_state(self, state):
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)
    return encoded_state

env = TicTacToe()
state = env.get_initial_state()
state = env.get_next_state(state, 0, -1)
state = env.get_next_state(state, 1, 1)
print(state)
encoded_state = env.get_encoded_state(state)
print(encoded_state)

tensor_state = tf.convert_to_tensor(encoded_state)
tensor_state = tf.expand_dims(tensor_state, axis=0)
print(tensor_state)