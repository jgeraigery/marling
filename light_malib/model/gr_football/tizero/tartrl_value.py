# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from threading import local
import numpy as np
import gym

from light_malib.utils.logger import Logger

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .tartrl_common import RNNLayer, FcEncoder, check, init, get_fc
from .positional_encoding import PositionalEncoding

from ....utils.running_stats import RunningStats

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class ValueInputEncoder(nn.Module):
    def __init__(self, n_agents):
        super(ValueInputEncoder, self).__init__()
        fc_layer_num = 2
        fc_output_num = 64

        self.n_agents = n_agents
        self.ball_input_num = 12
        if self.n_agents == 10:
            self.ball_owner_input_num = 23
            self.left_input_num = 88
            self.right_input_num = 88
        elif self.n_agents == 5:
            self.ball_owner_input_num = 13
            self.left_input_num = 48
            self.right_input_num = 48
        elif self.n_agents == 4:
            self.ball_owner_input_num = 11
            self.left_input_num = 40
            self.right_input_num = 40
        elif self.n_agents == 3:
            self.ball_owner_input_num = 9
            self.left_input_num = 32
            self.right_input_num = 32      
        self.match_state_input_num = 9

        self.ball_encoder = FcEncoder(fc_layer_num, self.ball_input_num, fc_output_num)
        self.ball_owner_encoder = FcEncoder(fc_layer_num, self.ball_owner_input_num, fc_output_num)
        self.left_encoder = FcEncoder(fc_layer_num, self.left_input_num, fc_output_num)
        self.right_encoder = FcEncoder(fc_layer_num, self.right_input_num, fc_output_num)
        self.match_state_encoder = FcEncoder(fc_layer_num, self.match_state_input_num, self.match_state_input_num) # TODO: suspicious

    def forward(self, x):

        ball_vec = x[:, :self.ball_input_num]  
        ball_owner_vec = x[:, self.ball_input_num : self.ball_input_num + self.ball_owner_input_num]  
        left_vec = x[:, self.ball_input_num + self.ball_owner_input_num : self.ball_input_num + self.ball_owner_input_num + self.left_input_num]  
        right_vec = x[:, self.ball_input_num + self.ball_owner_input_num + self.left_input_num : \
            self.ball_input_num + self.ball_owner_input_num + self.left_input_num + self.right_input_num]  
        match_state_vec = x[:, self.ball_input_num + self.ball_owner_input_num + self.left_input_num + self.right_input_num:]
    
        ball_output = self.ball_encoder(ball_vec)
        ball_owner_output = self.ball_owner_encoder(ball_owner_vec)  
        left_output = self.left_encoder(left_vec)  
        right_output = self.right_encoder(right_vec)  
        match_state_output = self.match_state_encoder(match_state_vec)  
    
        return torch.cat([  
            ball_output,  
            ball_owner_output,  
            left_output,  
            right_output,  
            match_state_output
        ], 1)  


def get_mlp(input_size, output_size, num_layers):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)

class ValueObsEncoder(nn.Module):
    def __init__(self, input_embedding_size, hidden_size, _recurrent_N, _use_orthogonal, rnn_type, n_agents):
        super(ValueObsEncoder, self).__init__()
        self.input_encoder = ValueInputEncoder(n_agents)     
        self.input_embedding = get_fc(input_embedding_size, hidden_size)
        # TODO: bring back the LSTM, after fixing it
        # self.rnn = RNNLayer(hidden_size, hidden_size, _recurrent_N, _use_orthogonal, rnn_type=rnn_type)  
        # self.after_rnn_mlp = nn.Linear(hidden_size, hidden_size) 
        self.last_mlp = get_mlp(hidden_size, hidden_size, 5) 

    def forward(self, obs, rnn_states, masks):
        actor_features = self.input_encoder(obs)
        actor_features = self.input_embedding(actor_features)
        # output, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # output = self.after_rnn_mlp(output)
        output = self.last_mlp(actor_features)
        return output, rnn_states

class ValueNetwork(nn.Module):
    def __init__(self, n_agents, device=torch.device("cpu")):
        super(ValueNetwork, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.hidden_size = 256
        recurrent_N = 1
        use_orthogonal = True
        rnn_type = 'lstm'
        input_embedding_size = 64 * 4 + 9
        self.n_agents = n_agents

        self.running_stats = RunningStats(device=device) 
        self.obs_encoder = ValueObsEncoder(input_embedding_size, self.hidden_size, recurrent_N, use_orthogonal, rnn_type, self.n_agents)
        self.final_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.LayerNorm(self.hidden_size), 
                                         nn.Linear(self.hidden_size, 1))
    
    def forward(self, obs, rnn_states, masks=np.concatenate(np.ones((1, 1, 1), dtype=np.float32))):
        obs = check(obs).to(dtype=torch.float32, device=self.device)

        # Reshape the masks to be only one, as we only have on RNN state
        # if masks.shape[0] != 10:
        # 
        #
        # if len(obs.shape) > 2:
        #     obs = obs.view(-1, self.n_agents, obs.shape[-1])[:, 0, :].view(1, -1)
        # else:
        #     obs = obs[0, :].view(1, -1)

        # Create a set of 1 masks with the same dimension as the observation
        # masks = np.ones((obs.shape[0], 1), dtype=np.float32)

        masks = check(masks).to(dtype=torch.float32, device=self.device)
        rnn_states = check(rnn_states).to(dtype=torch.float32, device=self.device)

        obs_encodings, rnn_states = self.obs_encoder(obs, rnn_states, masks)
        values = self.final_encoder(obs_encodings)
        
        return values, rnn_states
    
    def to_device(self, device):
        self.device = device