# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from turtle import forward
from matplotlib.pylab import f
import torch
import torch.nn as nn  
import torch.optim as optim  
  
import numpy as np

from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger

# Network for the intrinsic reward term (R_phi)
class IntrinsicRewardNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, n_nodes=128, gain=1., device="cuda"):
        super().__init__()

        self.device = device
        self.gain = gain

        layers = []
        layers.append(nn.Linear(input_dim, n_nodes))
        layers.append(nn.ReLU())

        self.output_dim = output_dim
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_nodes, n_nodes))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(n_nodes, output_dim))
        self.network = nn.Sequential(*layers).float()

        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

        self.to(self.device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=self.gain)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def reset_hidden(self, batch_size):
        #self.gru_hidden = torch.zeros(1, batch_size, self.output_dim).to(self.device)
        pass

    def forward(self, x):
        # n_batch, n_players, n_features = x.size()
        # x = x.reshape(n_batch * n_players, n_features)  # combine n_batch and n_players into a single dimension

        # x, hn = self.gru(x, self.gru_hidden)  # now x should have 3 dimensions (seq_len, batch, input_size)
        # self.gru_hidden = hn  

        # x = x[:, -1, :]
        # x = x.reshape(n_batch, n_players, 19)  # separate n_players and n_outputs dimensions
        x = self.network(x) 
        x = self.tanh(x)

        # x = x.reshape(n_batch, n_players, -1)

        return x  

# Intrinsic reward term (R_phi)
class IntrinsicReward:
    def __init__(self, network, learning_rate=5.e-4, n_actions=19, clipping_epsilon=0.1, device="cuda"):
        self.network = network
        self.device = device
        self.clipping_epsilon = clipping_epsilon
        self.n_actions = n_actions

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def get_intrinsic_reward(self, states, actions):
        intrinsic_rew = self.network(states)
        actions = actions.squeeze(-1).long()  # removes the last dimension from actions  
        new_rew = intrinsic_rew.gather(-1, actions.unsqueeze(-1)) * self.clipping_epsilon  
        new_rew = new_rew.squeeze(-1)
        return new_rew    

    def reset_hidden(self, batch_size):
        #self.network.module.reset_hidden(batch_size)
        pass

    def update_intrinsic_reward(self, sample, action_logs_prob, agent_num):  
        actions_batch = sample[EpisodeKey.ACTION].long()

        if agent_num == 10:
            states_batch = sample[EpisodeKey.CUR_STATE][:, 19:330+19]
        elif agent_num == 5:
            states_batch = sample[EpisodeKey.CUR_STATE][:, 19:210+19]

        G_bar_batch = sample[EpisodeKey.RETURN]
        V_s = sample[EpisodeKey.STATE_VALUE]

        pi_given_s_array_batch = action_logs_prob.exp()[:, :-1]  

        # Reset the hidden state of the network
        # self.reset_hidden(sequences.size(0))
        network_outputs = self.network(states_batch)

        # Flatten the first two dimensions 
        # network_outputs = network_outputs.reshape(-1, self.n_actions)

        a_term = network_outputs.gather(-1, actions_batch.unsqueeze(-1))

        b_term = network_outputs * pi_given_s_array_batch

        # b_term = torch.sum(b_term, dim=-1, keepdim=True)

        final_result_left_hand_side = torch.sum(a_term, dim=-1) - torch.sum(b_term, dim=-1)

        tmp1 = G_bar_batch - V_s

        tmp1 = tmp1.unsqueeze(-1).repeat(1, agent_num).view(-1, 1)

        tmp2 = tmp1 * final_result_left_hand_side.unsqueeze(-1)

        pi_per_s = pi_given_s_array_batch.gather(-1, actions_batch.unsqueeze(-1))

        all_l_terms = pi_per_s * tmp2

        loss = -torch.mean(all_l_terms)

        # update SelfRS network  
        self.network.zero_grad() 
        self.optimizer.zero_grad()  
        loss.backward(retain_graph=True)  
        self.optimizer.step()  