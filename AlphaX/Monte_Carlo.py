import numpy as np
import tensorflow as tf

class Node:
    def __init__(self, args, state, game, parent=None, action_taken_from_parent=None, possible_actions=None):
        self.args = args
        self.state = state
        self.game = game
        self.parent = parent
        self.action_taken_from_parent = action_taken_from_parent
        self.untried_actions = possible_actions if possible_actions is not None else game.get_valid_moves(state)
        self.children = []
        self.visit_count = 0
        self.value_sum = 0

    def select(self, state):
        best_child = None
        best_value = -np.inf

        for child in self.children:
            q_value = 1 - ((child.value_sum() / child.visit_count) + 1) / 2
            uct_value = q_value + self.args['C'] * np.sqrt(np.log(self.visit_count) / child.number_of_visits)

            if uct_value > best_value:
                best_value = uct_value
                best_child = child

        return best_child
    
    def add_child(self, child_node):
        self.children[child_node.action_taken_from_parent] = child_node

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0 and len(self.children) > 0
    
    def expand_node(self, untried_ctions: list):
        if self.parent is None or self.parent.is_fully_expanded:
            return self.parent

        if len(untried_ctions) > 0:
            action = np.random.choice(untried_ctions)
            untried_ctions.remove(action)

            new_state = self.game.get_next_state(self.parent.state, action, 1)
            child_node = Node(self.args, new_state, self.game, self, action, untried_ctions)
            self.add_child(child_node)
            return child_node

        return self.parent
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)

    def best_child(self):
        best_value = -1
        best_action = None

        for child in self.children:
            if self.children[child].visit_count > best_value:
                best_value = self.children[child].visit_count
                best_action = child
        
        return best_action
    
class MCTS:
    def __init__(self, args, game, model):
        self.args = args
        self.game = game
        self.model = model
    
    def search(self, state):
        root = Node(self.args, state, self.game)

        for i in range(self.args['num_searches']):

            node = root
            while not node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_game_value(node.state, node.action_taken_from_parent)
            value = self.game.get_opponent_value(value)

            if is_terminal:
                policy, value = self.model(
                    tf.constant(tf.expand_dims(self.game.get_encoded_state(node.state), axis=0))
                )

                policy = tf.nn.softmax(policy).numpy()[0]
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.numpy()
                node.expand_node(policy)

            node.backpropagate(value)

        action_probs = np.zeros(shape=(3 * 3))

        for child in root.children:
            action_probs[child] = root.children[child].visit_count

        action_probs /= np.sum(action_probs)
        return action_probs
