import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np 

class Network(nn.Module):
    def __init__(self, board_size):
        super(Network, self). __init__()

        self.board_size = board_size

        # common layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        # policy layers
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm1d(2 * self.board_size * self.board_size)
        self.policy_fc = nn.Linear(2 * self.board_size * self.board_size, self.board_size * self.board_size)

        # value layers
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm1d(self.board_size * self.board_size)
        self.value_fc1 = nn.Linear(self.board_size * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, state):
        # common layers
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # policy layers
        x_pol = self.policy_conv(x)
        x_pol = x_pol.view(-1, 2 * self.board_size * self.board_size)
        x_pol = F.relu(self.policy_bn(x_pol))
        x_pol = self.policy_fc(x_pol)

        # value layers
        x_val = self.value_conv(x)
        x_val = x_val.view(-1, self.board_size * self.board_size)
        x_val = F.relu(self.value_bn(x_val))
        x_val = F.relu(self.value_fc1(x_val))
        x_val = torch.tanh(self.value_fc2(x_val))

        return x_pol, x_val

def set_learning_rate(optimizer, lr):
    """
    set learning rate to the given value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class PolicyValueNet_Operation:
    def __init__(self, board_size, model_path=None):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PolicyValueNet = Network(self.board_size).to(self.device)
        self.l2_reg = 1e-4
        self.optimizer = optim.SGD(self.PolicyValueNet.parameters(), lr = 0.02, momentum=0.9)
        if model_path:
            net_parameter = torch.load(model_path)
            self.PolicyValueNet.load_state_dict(net_parameter)

    def save_parameter(self, model_path):
        net_parameter = self.PolicyValueNet.state_dict()
        torch.save(net_parameter, model_path)
    
    def get_probs_value(self, state):
        """
        get probability and value of the given state

        input:
          state -- numpy array of shape (board_size, board_size, 3)

        output:
          probs -- probability of actions, numpy array of shape (board_size * board_size)
          value -- value of current state, scalar
        """
        self.PolicyValueNet.eval()
        state = torch.FloatTensor([state]).to(self.device)

        policy_output, value = self.PolicyValueNet(state) # outputs (1, board_size * board_size), (1,1)
        probs = F.softmax(policy_output.cpu().view(-1), dim=-1).detach().numpy()

        return probs, value

    def train_step(self, batch_state, batch_mcts_prob, batch_reward, lr=None):
        """
        train network for one minibatch

        input:
          batch_state -- minibatch of states, numpy array of shape (batch_size, board_size, board_size, 3)
          batch_mcts_prob -- probability of actions by mcts, numpy array of shape (batch_size, board_size * board_size)
          batch_reward -- reward of game, numpy array of shape (batch_size)
          lr -- learning rate

        output:
          loss -- for monitoring
        """
        self.PolicyValueNet.train()
        batch_state = torch.FloatTensor(batch_state).to(self.device)
        batch_mcts_prob = torch.FloatTensor(batch_mcts_prob).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).to(self.device)

        # set optimizer
        self.optimizer.zero_grad()
        if lr != None:
          set_learning_rate(self.optimizer, lr)

        # forward
        policy_output, value = self.PolicyValueNet(batch_state)
        log_action_prob = F.log_softmax(policy_output, dim=-1)

        # define loss
        value_loss = F.mse_loss(value.view(-1), batch_reward)
        policy_loss = -torch.mean(torch.sum(batch_mcts_prob * log_action_prob, 1))
        loss = value_loss + policy_loss

        # optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()
    




