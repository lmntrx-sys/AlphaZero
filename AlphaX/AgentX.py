from Game import TicTacToe
from model import resnet_tictactoe_model
from Monte_Carlo import MCTS
import random

import tensorlfow as tf # type: ignore
import numpy as np

class AlphaZero:
    def __init__(self, game, args, optimizer, model):
        self.game= game
        self.args = args
        self.optimizer = optimizer
        self.model = model
        self.mcts = MCTS(self.args, self.game, self.model)

    def self_paly(self):
        state = self.game.get_initial_state()
        player = 1
        memory = []

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_space(), p=action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_game_value()

            if is_terminal:
                returnMemory = []

                for hist_action_probs, hist_neutral_state, cplayer in memory:
                    hist_outcome = value if cplayer == player else self.game.get_opponent_value(value)
                    returnMemory = returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs, 
                        hist_outcome
                    ))

                return returnMemory
            player = self.game.get_oppponent(player)

    def train(self, memory):

        random.shuffle(memory)

        for batch_idx in range(0, len(memory), self.args['batch_size']):

            sample = memory[batch_idx:min(len(memory), batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            state = tf.constant(state, dtype=tf.float32) 
            policy_targets = tf.constant(policy_targets, dtype=tf.float32) 
            value_targets = tf.constant(value_targets, dtype=tf.float32)

            with tf.GradientTape() as tape: 
                policy_preds, value_preds = self.model(state)
                policy_loss = tf.keras.losses.CategoricalCrossentropy()(policy_targets, policy_preds)
                value_loss = tf.keras.losses.MeanSquaredError()(value_targets, value_preds)
                total_loss = policy_loss + value_loss
                print(f'policy loss: {policy_loss.numpy()} value loss: {value_loss.numpy()} total loss: {total_loss}')

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
    
    def learn(self):

        for iter in range(self.args['num_iterations']):
            memory = []

            for selfPlay in range(self.args['num_selfPlay_iterations']):
                memory += selfPlay

            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            model_path = './model.h5'

        # Save states
        model_path = f'./model_iter_{iter+1}.h5'
        tf.keras.models.save_model(self.model, filepath=model_path)
