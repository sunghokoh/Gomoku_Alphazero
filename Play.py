import pygame
import numpy as np
from Model import PolicyValueNet_Operation
from Game import Gomoku
from MCTS import MCTS

class PlayGame:
    def __init__(self, model_path):
        pygame.init()

        # game setting
        self.board_size = 6
        self.n_simulation = 400
        self.human_turn = 1
        self.AI_turn = -1

        self.Game = Gomoku(self.board_size)
        self.Network = PolicyValueNet_Operation(self.board_size, model_path)
        self.MCTS = MCTS(self.Network)
        _ = self.MCTS.get_mcts_probs(self.Game, 1.0, 100)

        self.last_action = None
        self.winner = 0
        self.finished = False

        self.init_window()

    
    def init_window(self):
        # setting
        self.boardColor = (244, 164, 96)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.spaceSize = 40
        self.stoneSize = self.spaceSize / 4
        self.monitor_height = 0
        self.gridwidth = 2
        self.board_length = (self.board_size + 1) * self.spaceSize
        print(self.board_length)
        
        self.window = pygame.display.set_mode((self.board_length, self.board_length))

    def draw_windwow(self):
        self.window.fill(self.boardColor)

        for i in range(1, self.board_size + 1):
            pygame.draw.line(self.window, self.black, (i * self.spaceSize, self.spaceSize), (i * self.spaceSize, self.board_length - self.spaceSize), self.gridwidth)
            pygame.draw.line(self.window, self.black, (self.spaceSize, i * self.spaceSize), (self.board_length - self.spaceSize, i * self.spaceSize), self.gridwidth)
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                coor = self.boardPos_to_windowCoor((r,c))

                if self.Game.board[r, c] == 1:
                    pygame.draw.circle(self.window, self.black, coor, int(self.stoneSize))
                if self.Game.board[r, c] == -1:
                    pygame.draw.circle(self.window, self.white, coor, int(self.stoneSize))

        pygame.display.update()


    def boardPos_to_windowCoor(self, pos):
        """
        convert board position to window Coordinate

        input:
          pos -- tuple of shape (row, column)
        """
        r, c = pos
        x = (c + 1) * self.spaceSize
        y = (r + 1) * self.spaceSize

        return (x, y)
    
    def human_action(self):
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()

            x_remain = x % self.spaceSize
            y_remain = y % self.spaceSize
            
            if x_remain < self.stoneSize:
                c = x // self.spaceSize -1
            elif x_remain > self.spaceSize - self.stoneSize:
                c = x // self.spaceSize 
            else :
                c = None

            if y < self.board_length:
                if y_remain < self.stoneSize:
                    r = y // self.spaceSize -1
                elif y_remain > self.spaceSize - self.stoneSize:
                    r = y // self.spaceSize
                else : 
                    r = None
            else:
                r = None
            
            if r != None and c != None:
                if self.Game.board[r, c] == 0:
                    action = r * self.board_size + c 
                    print(r, c)
                    self.last_action = action
                    _, self.winner, self.finished = self.Game.step(action)
    
    def AI_action(self):
        self.MCTS.change_root(self.last_action)

        mcts_probs = self.MCTS.get_mcts_probs(self.Game, 1.0, self.n_simulation)

        action = np.argmax(mcts_probs)
        _, self.winner, self.finished = self.Game.step(action)


Play = PlayGame(model_path=None)

Run = True
while Run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Run = False

    if Play.finished==False and Play.Game.player == Play.human_turn:
        Play.human_action()
    
    elif Play.finished==False and Play.Game.player == Play.AI_turn:
        Play.AI_action()
    
    Play.draw_windwow()
    
    

