from sample_players import DataPlayer
import numpy as np
import random

# This Monte-Carlo Tree Search was implemented using the following article as a reference:
# https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************

    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.terminal_test() or state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            root_node = MCTSNode(state)
            while (True):
                if root_node.state.terminal_test():
                    return random.choice(state.actions())
                child = selection(root_node)
                reward = simulation(child.state)
                backpropagation(child, reward)
                index = root_node.children.index(best_ucb(root_node))
                best_action = root_node.children_actions[index]
                self.queue.put(best_action)

# The selection step traverses (by expanding) the current tree and uses the "best_ucb" algorithm
# to select the best node based on the Upper-Confidence-Bound (UCB)
def selection(node):
    while not node.state.terminal_test():
        if not len(node.children_actions) == len(node.state.actions()):
            return expansion(node)
        node = best_ucb(node)
    return node


# The expansion step adds new child node to the tree
def expansion(node):
    for action in node.state.actions():
        if action not in node.children_actions:
            not_tried_state = node.state.result(action)
            node.add_child_node(not_tried_state, action)
            node_child = node.children[-1]
            return node_child


# The simulation step randomly chooses available moves until the game is over
def simulation(state):
    player_copy = state.player()
    while not state.terminal_test():
        action = random.choice(state.actions())
        state = state.result(action)
    if state._has_liberties(player_copy):
        return -1
    else:
        return 1


# The backpropagation step recursively updates all the statistics and rewards the "good"
# nodes/actions
def backpropagation(node, reward):
    node.reward += reward
    node.visits += 1
    reward = -reward
    if node.parent:
        backpropagation(node.parent, reward)


# The MCTS node structure
class MCTSNode():
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.reward = 0
        self.children = []
        self.children_actions = []

    def add_child_node(self, state, action):
        child = MCTSNode(state, self)
        self.children.append(child)
        self.children_actions.append(action)


# This implements the UCB to return the weight
def best_ucb(node):
    values = [
        (child.reward / child.visits) + np.sqrt(2. * np.log(node.visits) / child.visits)
        for child in node.children
    ]

    return node.children[np.argmax(values)]
