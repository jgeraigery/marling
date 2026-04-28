#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified portions of the file are Copyright (c) 2026 Electronic Arts Inc.

# Copyright 2023 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from cv2 import log
import numpy as np
import gym

import torch
import torch.nn as nn
from torch.distributions import Categorical

from light_malib.utils.logger import Logger
from .tartrl_common import RNNLayer, FcEncoder, check, init, get_fc
from .positional_encoding import PositionalEncoding

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        super_logprobs = super().log_prob(actions.squeeze(-1))
        new_lps = super_logprobs.view(actions.size(0), -1)
        logprobs = new_lps.sum(-1).unsqueeze(-1)
        return logprobs
    
    def log_probs_all(self):
        return super().logits

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
            available_actions = available_actions.reshape(x.shape)
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.multidiscrete_action = False
        self.continuous_action = False
        self.mixed_action = False

        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False, actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        if actions is None:
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            dist_entropy = None
        else:
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy()
        action_log_probs = action_logits.log_probs(actions)

        # Get the log probability of all actions as well for intrinsic rewrad shaping
        action_log_probs_all = action_logits.log_probs_all()

        return actions, action_log_probs, dist_entropy, action_log_probs_all

class PolicyInputEncoder(nn.Module):
    def __init__(self, n_agents):
        super(PolicyInputEncoder, self).__init__()
        fc_layer_num = 2
        fc_output_num = 64
        self.n_agents = n_agents

        if self.n_agents == 10:
            self.active_input_num = 87
            self.ball_owner_input_num = 57
            self.left_input_num = 88
            self.right_input_num = 88
        elif self.n_agents == 5:
            self.active_input_num = 57
            self.ball_owner_input_num = 47
            self.left_input_num = 48
            self.right_input_num = 48
        elif self.n_agents == 4:
            self.active_input_num = 51
            self.ball_owner_input_num = 45
            self.left_input_num = 40
            self.right_input_num = 40
        elif self.n_agents == 3:
            self.active_input_num = 45
            self.ball_owner_input_num = 43
            self.left_input_num = 32
            self.right_input_num = 32
        
        self.match_state_input_num = 9

        self.active_encoder = FcEncoder(fc_layer_num, self.active_input_num, fc_output_num)
        self.ball_owner_encoder = FcEncoder(fc_layer_num, self.ball_owner_input_num, fc_output_num)
        self.left_encoder = FcEncoder(fc_layer_num, self.left_input_num, fc_output_num)
        self.right_encoder = FcEncoder(fc_layer_num, self.right_input_num, fc_output_num)
        self.match_state_encoder = FcEncoder(fc_layer_num, self.match_state_input_num, self.match_state_input_num)

    def forward(self, x):

        active_vec = x[:, :self.active_input_num]
        ball_owner_vec = x[:, self.active_input_num : self.active_input_num + self.ball_owner_input_num]
        left_vec = x[:, self.active_input_num + self.ball_owner_input_num : self.active_input_num + self.ball_owner_input_num + self.left_input_num]
        right_vec = x[:, self.active_input_num + self.ball_owner_input_num + self.left_input_num : \
            self.active_input_num + self.ball_owner_input_num + self.left_input_num + self.right_input_num]
        match_state_vec = x[:, self.active_input_num + self.ball_owner_input_num + self.left_input_num + self.right_input_num:]

        active_output = self.active_encoder(active_vec)
        ball_owner_output = self.ball_owner_encoder(ball_owner_vec)
        left_output = self.left_encoder(left_vec)
        right_output = self.right_encoder(right_vec)
        match_state_output = self.match_state_encoder(match_state_vec)

        return torch.cat([
            active_output,
            ball_owner_output,
            left_output,
            right_output,
            match_state_output
        ], 1)

def get_fc(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size))

def get_mlp(input_size, output_size, num_layers):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)

class PolicyObsEncoder(nn.Module):
    def __init__(self, input_embedding_size, hidden_size, _recurrent_N, _use_orthogonal, rnn_type, n_agents):
        super(PolicyObsEncoder, self).__init__()
        self.input_encoder = PolicyInputEncoder(n_agents)
        self.input_embedding = get_fc(input_embedding_size, hidden_size)
        # self.rnn = RNNLayer(hidden_size, hidden_size, _recurrent_N, _use_orthogonal, rnn_type=rnn_type) 
        # self.after_rnn_mlp = get_fc(hidden_size, hidden_size) 
        self.last_mlp = get_mlp(hidden_size, hidden_size, 5)

    def forward(self, obs, rnn_states, masks):
        actor_features = self.input_encoder(obs)
        actor_features = self.input_embedding(actor_features)
        # TODO: Bring back the LSTM, after fixing it
        # output, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # output = self.after_rnn_mlp(actor_features)
        output = self.last_mlp(actor_features)

        return output, rnn_states


class PolicyNetwork(nn.Module):
    def __init__(self, n_agents, use_pos_encoding=False, device=torch.device("cpu")):
        super(PolicyNetwork, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.hidden_size = 256
        self._use_policy_active_masks = True
        recurrent_N = 1
        use_orthogonal = True
        rnn_type = 'lstm'
        gain = 0.01
        action_space = gym.spaces.Discrete(20)
        self.action_dim = 19
        input_embedding_size = 64 * 4 + 9
        self.active_id_size = 1
        self.id_max = 10
        self.n_agents = n_agents

        self.obs_encoder = PolicyObsEncoder(input_embedding_size, self.hidden_size, recurrent_N, use_orthogonal, rnn_type, self.n_agents)

        Logger.info(f'Using Positional Encoding for Player ID: {use_pos_encoding}')
        self.use_pos_encoding = use_pos_encoding

        if self.use_pos_encoding:
            # Use Positional Encoding to encode the player IDs directly (i.e. not one-hot encoded),
            # which is different from before where a fully-connected layer was used on one-hot ID.
            d_model = self.hidden_size
            self.id_embedding = PositionalEncoding(d_model, max_len=5000)

            self.before_act_wrapper = FcEncoder(2, self.hidden_size + d_model, self.hidden_size)
        
        else:
            self.id_max = 11
            # Use a fully-connected layer to encode the player IDs
            self.id_embedding = get_fc(self.id_max, self.id_max)

            self.before_act_wrapper = FcEncoder(2, self.hidden_size + self.id_max, self.hidden_size)

        self.act = ACTLayer(action_space, self.hidden_size, use_orthogonal, gain)

        self.to(device)


    def forward(self, obs, rnn_states, actions=None, active_masks=None, masks=np.concatenate(np.ones((1, 1, 1), dtype=np.float32)), available_actions=None, deterministic=False):
        obs = check(obs).to(dtype=torch.float32, device=self.device)
        if available_actions is not None:
            available_actions = check(available_actions).to(dtype=torch.float32, device=self.device)
        masks = check(masks).to(dtype=torch.float32, device=self.device)
        rnn_states = check(rnn_states).to(dtype=torch.float32, device=self.device)
        if actions is not None:
            actions = check(actions).to(dtype=torch.float32, device=self.device)
        if active_masks is not None:
            active_masks = check(active_masks).to(dtype=torch.float32, device=self.device)

        active_id = obs[:,:self.active_id_size].squeeze(1).long()

        if self.use_pos_encoding:
            # Perform Positional Encoding on the active_id, without using one-hot encoding as before
            id_output = self.id_embedding(active_id)
        else:
            # Use one-hot encoding and fully-connected layer
            id_onehot = torch.eye(self.id_max, device=self.device)[active_id].to(self.device)
            id_output = self.id_embedding(id_onehot)

        obs = obs[:,self.active_id_size:]
        obs_output, rnn_states = self.obs_encoder(obs, rnn_states, masks)

        # Concatenate the positional encoding and the observation encoding
        output = torch.cat([id_output, obs_output], 1)
        # Sum up the positional encoding and the observation encoding
        #output = id_output + obs_output

        output = self.before_act_wrapper(output)

        actions, action_log_probs, dist_entropy, action_log_probs_all = self.act(output, available_actions, deterministic=deterministic, actions=actions, active_masks=active_masks)
        return actions, rnn_states, action_log_probs, dist_entropy, action_log_probs_all
    
    def to_device(self, device):
        self.device = device
    

