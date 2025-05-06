from Game import TicTacToe
from model import resnet_tictactoe_model
from Monte_Carlo import MCTS
import random

import tensorlfow as tf # type: ignore
import numpy as np

class AlphaZero:
  def __init__(self, model, optimizer, game, args):
    self.model = model
    self.optimizer = optimizer
    self.game = game
    self.args = args
    self.mcts = MCTS(game, args, model)
    self.step = tf.Variable(0, dtype=tf.int64)

    # --- TensorBoard Setup ---
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    self.summary_writer = tf.summary.create_file_writer(log_dir)
    self.log_dir = log_dir
    # -------------------------


  def selfPlay(self):
    memory = []
    player = 1
    state = self.game.get_initial_state()

    while True:
      neutral_state = self.game.change_perspective(state, player)
      action_probs = self.mcts.search(neutral_state)

      memory.append((neutral_state, action_probs, player))

      action = np.random.choice(self.game.action_size, p=action_probs)

      state = self.game.get_next_state(state, action, player)

      value, is_terminal = self.game.get_value_and_terminated(state, action)

      if is_terminal:
        returnMemory = []
        for hist_neutral_state, hist_action_probs, hist_player in memory:
          hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
          returnMemory.append((
            self.game.get_encoded_state(hist_neutral_state),
            hist_action_probs,
            hist_outcome
          ))
        return returnMemory

      player = self.game.get_opponent(player)

  @tf.function
  def train(self, dataset):
    for state, policy_targets, value_targets in dataset:
      # ---------------------------------CHECK------------------------------------------------------

      self.step += 1

      with tf.GradientTape() as tape:
        policy_preds, value_preds = self.model(state)
        policy_loss = tf.keras.losses.CategoricalCrossentropy()(policy_targets, policy_preds)
        value_loss = tf.keras.losses.MeanSquaredError()(value_targets, value_preds)

        total_loss = policy_loss + value_loss

      grads = tape.gradient(total_loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

      # --- Log to TensorBoard ---
      with self.summary_writer.as_default():
          tf.summary.scalar('policy_loss', policy_loss, step=self.step)
          tf.summary.scalar('value_loss', value_loss, step=self.step)
          tf.summary.scalar('total_loss', total_loss, step=self.step)
      # -------------------------

    #---------------------------------CHECK------------------------------------------------------------

  def learn(self):

    if not hasattr(self, 'summary_writer'):
      log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
      self.summary_writer = tf.summary.create_file_writer(log_dir)


    for iteration in range(self.args['num_iterations']):
      memory = []

      for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
        memory += self.selfPlay()

      dataset = tf.data.Dataset.from_tensor_slices((
          [mem[0] for mem in memory],
          [mem[1] for mem in memory],
          [mem[2] for mem in memory]
      ))
      dataset = dataset.shuffle(buffer_size=max(1, len(memory))).batch(self.args['batch_size'])

      #self.model.train()
      for epoch in trange(self.args['num_epochs']):
        self.train(dataset)

    model_path = './content/drive/My Drive/model.h5'# <--- CHECK
    tf.keras.models.save_model(self.model, model_path)# <--- CHECK
