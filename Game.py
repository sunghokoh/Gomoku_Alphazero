import numpy as np

class Gomoku:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size)).astype(int)
        self.legal_actions = np.ones(self.board_size * self.board_size).astype(int)
        self.player = 1 # black == 1, white == -1
    
    def reset(self):
        """
        reset game
        
        output:
         observation -- board state 
        """
        self.board = np.zeros((self.board_size, self.board_size)).astype(int)
        self.legal_actions = np.ones(self.board_size * self.board_size).astype(int)
        self.player = 1 # black == 1, white == -1
        
        return self.get_observation()

    def get_legal_actions(self):
        """
        return legal actions

        output:
          legal_actions -- list of legal actions 
        """
        return self.legal_actions.astype(float)

    def is_finished(self, location):
        """
        check game is finished

        input:
          loacation -- latest action, (row, column)

        output:
          winner -- 1 : black, -1 : white, 0 : draw or not terminate
          finished -- True of False
        """
        # check draw
        if self.legal_actions.sum() == 0: # board is full
            return 0, True
        
        # check winner
        r, c = location
        direction = ((0, 1), (1, 1), (1, 0), (1, -1))

        for d in direction:
            count = 1
            for i in [1, -1]: # also check the opposite direction
                x, y = r, c
                for _ in range(5):
                    x += (i * d[0])
                    y += (i * d[1])
                    
                    if (x not in range(self.board_size)) or (y not in range(self.board_size)):
                        break
                    if self.board[x, y] != self.player:
                        break
                    count += 1

                    if count == 5: # location counted twice
                        return self.player, True
        
        return 0, False

    def get_observation(self):
        """
        return observation of current board

        output:
          observation -- numpy array of shape [board_current_player, board_opponent_player, board_player], board_player has a constant value of either 1 if black is to play or -1 if white is to play
        """
        board_black = np.where(self.board == 1, 1.0, 0.0)
        board_white = np.where(self.board == -1, 1.0, 0.0)
        board_player = np.full((self.board_size, self.board_size), self.player).astype(float)
        
        if self.player == 1:
            return np.array([board_black, board_white, board_player])
        elif self.player == -1:
            return np.array([board_white, board_black, board_player])


    def step(self, action):
        r = action // self.board_size
        c = action % self.board_size
        self.board[r, c] = self.player
        self.legal_actions[action] = 0

        winner, finished = self.is_finished((r, c))

        self.player *= -1

        return self.get_observation(), winner, finished
        

    