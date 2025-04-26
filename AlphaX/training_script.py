from Game import TicTacToe
from AgentX import AlphaZero
from model import resnet_tictactoe_policy


import tensorflow as tf


tictactoe = TicTacToe()
player = 1

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

args = {
'C': 2,
'num_searches': 50,
'num_iterations': 4,
'num_selfPlay_iterations': 100,
'num_epochs': 5,
'batch_size': 64
}


model = resnet_tictactoe_policy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
alphaZero = AlphaZero(model, optimizer, tictactoe, args)
alphaZero.learn()
