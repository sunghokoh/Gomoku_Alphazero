import random
import numpy as np
from Game import Gomoku
from Model import PolicyValueNet_Operation
from MCTS import MCTS

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, data):
        """
        push data to the memory

        input:
          data -- (state, mcts_prob, reward)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(data)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)

        batch_state = []
        batch_mcts_prob = []
        batch_reward = []

        for data in batch_data:
            batch_state.append(data[0])
            batch_mcts_prob.append(data[1])
            batch_reward.append(data[2])

        batch_state = np.array(batch_state)
        batch_mcts_prob = np.array(batch_mcts_prob)
        batch_reward = np.array(batch_reward)

        return batch_state, batch_mcts_prob, batch_reward

    def __len__(self):
        return len(self.memory)

class TrainPipeline:
    def __init__(self, model_path = None):
        # settings
        self.board_size = 6
        self.learning_rate = 0.02
        self.temp = 1.0
        self.n_simulation = 400
        self.buffer_size = 1000
        self.batch_size = 128
        self.epochs = 5
        self.n_game = 1500
        
        self.game = Gomoku(self.board_size)
        self.Network = PolicyValueNet_Operation(self.board_size, model_path)
        self.MCTS = MCTS(self.Network)
        self.buffer = ReplayMemory(self.buffer_size)
    
    def make_self_play_data(self):
        """
        Self-play a game and push the data in buffer
        """
        data_lst = []
        game_step = 0

        # reset
        obs = self.game.reset()
        self.MCTS.reset_Tree()

        # self play
        finished = False
        while finished == False:
            mcts_prob = self.MCTS.get_mcts_probs(self.game, self.temp, self.n_simulation)
            action = random.choices(range(len(mcts_prob)), weights=mcts_prob)[0]

            # save
            data_lst.append([obs, mcts_prob, self.game.player])
            
            # do action
            obs, winner, finished = self.game.step(action)
            self.MCTS.change_root(action)
            game_step += 1
        
        for data in data_lst:
            if winner == 0:
                data[2] = 0
            elif data[2] == winner:
                data[2] = 1
            else:
                data[2] = -1
            
            self.buffer.push(data)

        return game_step # for monitoring
    
    def train_Network(self):
        batch_state, batch_mcts_prob, batch_reward = self.buffer.sample(self.batch_size)
        loss_sum = 0.0
        for i in range(self.epochs):
            loss = self.Network.train_step(
                batch_state,
                batch_mcts_prob,
                batch_reward,
                lr = self.learning_rate)
            loss_sum += loss
        avg_loss = loss_sum/self.epochs
        return avg_loss
    
    def save(self):
         self.Network.save_parameter('./PolicyValue15.model')

    
    def run(self):
        for i in range(self.n_game):
            game_step = self.make_self_play_data()
            if self.buffer.__len__() > self.batch_size:
                loss = self.train_Network()
            else:
                loss = None
            
            print('game : {} data number : {}  game step : {}  loss = {}'.format(i+1, self.buffer.__len__(), game_step, loss))
            if i % 10 == 0 and i != 0 :
                self.save()
                print('model saved')
        self.save()
        print('model saved')

if __name__ == "__main__":    
    pipeline = TrainPipeline()
    pipeline.run()

