import numpy as np 
import copy

def softmax(x):
    probs = np.exp(x)
    probs /= np.sum(probs)
    return probs

class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = []
        self.prior_p = prior_p
        self.value_sum = 0
        self.num_visit = 0
        self.Q = 0

        self.c_puct = 5

    def expand(self, prior_action_p):
        """
        expand Tree by adding new childrens

        input:
          prior_action_p -- prior probability of all actions, numpy array of shape (num of actions)
        """
        self.children = [None for _ in range(len(prior_action_p))]
        for action, prob in enumerate(prior_action_p):
            self.children[action] = Node(self, prob)
    
    def select(self, legal_actions):
        """
        select action among children that haves maximum Q + U, does not select illegal moves
        """
        values = np.array([child.get_value() for child in self.children])
        values = values * legal_actions

        action = np.argmax(values)
        node = self.children[action]
        return action, node
    
    def get_value(self):
        """
        caculate Q + U of this node
        """
        Q = self.Q 
        U = self.c_puct * self.prior_p * np.sqrt(self.parent.num_visit) / (1 + self.num_visit)
        return Q + U

    def update(self, leaf_value):
        """
        update value sum and visit count

        input:
          leaf_value -- evaluated value of the leaf node
        """
        self.num_visit += 1
        self.value_sum += 1
        self.Q = self.value_sum / self.num_visit


    def is_root(self):
        return self.parent == None
    
    def is_leaf(self):
        return self.children == []

class MCTS:
    def __init__(self, PolicyValueNet):
        self.root = Node(None, 1.0)
        self.PolicyValueNet = PolicyValueNet

    def simulation(self, state):
        """
        Run single simulation of monte carlo tree search

        input:
          state -- copy of the enviroment at the root state
        """
        node = self.root

        # selection
        while True:
            if node.is_leaf():
                # add to prevent error at starting the game
                if node == self.root:
                    obs = state.get_observation()
                    finished = False
                    
                break

            action, node = node.select(state.get_legal_actions())
            obs, winner, finished = state.step(action)

        # evaluation
        action_probs, leaf_value = self.PolicyValueNet.get_probs_value(obs)
        if finished:
            if winner == 0: # draw
                leaf_value = 0.0
            else: 
                leaf_value = 1

        # expansion
        if not finished:
            node.expand(action_probs)

        # backup
        while node != self.root:
            leaf_value = -leaf_value
            node.update(leaf_value)
            node = node.parent


    def get_mcts_probs(self, state, temp, n_simulation):
        """
        run simulations and return action probability

        input:
          state -- current game state
          temp -- temperature parameter, contols exploration
          n_simulation -- number of simulation        
        """
        for n in range(n_simulation):
            state_copy = copy.deepcopy(state)
            self.simulation(state_copy)
        
        # calculate action probs
        action_visits = np.array([node.num_visit for node in self.root.children])
        legal_actions = state.get_legal_actions()
        action_visits = action_visits * legal_actions

        mcts_probs = softmax(1.0/temp * np.log(action_visits + 1e-10))

        return mcts_probs
    
    def change_root(self, action=None):
        """
        change root to self.root.children[action]
        """

        self.root = self.root.children[action]
        self.root.parent = None
    
    def reset_Tree(self):
        """
        reset Tree
        """
        self.root = Node(None, 1.0)