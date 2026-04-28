# Modified portions of the file are Copyright (c) 2026 Electronic Arts Inc. See LICENSE_lightmalib.md for details.

from threading import local
from tkinter import W
import torch
import torch.nn as nn
import numpy as np
import os
from gym.spaces import Box, Discrete

from light_malib.utils.logger import Logger

from .tartrl_value import ValueNetwork
from .tartrl_policy import PolicyNetwork
from .tartrl_utils import tartrl_obs_deal, _t2n, SPRINT, DRIBBLE, RELEASE_SPRINT

def check_tensor(data, device='cuda'):
    if not isinstance(data,torch.Tensor):
        data=torch.as_tensor(data, dtype=torch.float32, device=device)
    return data

#TODO: Modify both Actor and Critic to get the model configs from a given conf file (YAML).
# Currently the model configs are hardcoded.

class Actor(nn.Module):
    def __init__(
            self,
            model_config,
            observation_space,
            action_space,
            custom_config,
            initialization
    ):
        super().__init__()
        rnn_shape = [1, 1, 1, 512]
        self.rnn_layer_num = 1
        self.rnn_state_size = rnn_shape[-1]
        self.n_agents = int(custom_config["num_players_controlled"])
        self.rnn_hidden_state = [torch.zeros((1, rnn_shape[-1]), dtype=torch.float32) for _ in range(self.n_agents)]

        use_pos_encoding = model_config["use_pos_encoding"]
        self.model = PolicyNetwork(self.n_agents, use_pos_encoding=use_pos_encoding)

    def forward(self, obs, rnn_states, rnn_masks, action_masks, explore, given_actions, is_training=True):
        total_steps = int(obs.shape[0] / self.n_agents)  # Batch * Steps = Batch * Steps * N_agents / N_agents 

        # NOTE: Correct reshaping of obs
        # obs = obs.reshape((self.n_agents, total_steps, -1)) 
        obs = obs.reshape((total_steps, self.n_agents, -1))
        obs = obs.permute(1, 0, 2)

        rnn_states = rnn_states.reshape((total_steps, self.n_agents, -1))
        rnn_states = rnn_states.permute(1, 0, 2)

        rnn_masks = rnn_masks.reshape((self.n_agents, total_steps, -1))

        if given_actions is not None:
            # NOTE: Same reshaping scheme as obs
            given_actions = given_actions.reshape((total_steps, self.n_agents, -1))
            given_actions = given_actions.permute(1, 0, 2)
        
        action_list = []
        action_lps = []
        dist_entropies = []
        action_lps_all = []

        rnn_hidden_states = torch.zeros((self.n_agents, 1, 512), dtype=torch.float32)

        for i in range(obs.shape[0]):

            rnn_mask = rnn_masks[i]
            if given_actions is not None:
                current_actions = given_actions[i]
            else:
                current_actions = None

            # Getting the indiviual observations for each agent
            # NOTE: The correct order of the IDs is given by the correct reshaping of obs above
            each_obs = obs[i]
            if self.n_agents == 10:
                encoded_obs = each_obs[:, 19:330+19] # 330 Observations when 11v11 and controlling 10 players
            elif self.n_agents == 5:
                encoded_obs = each_obs[:, 19:210+19]
            elif self.n_agents == 4:
                encoded_obs = each_obs[:, 19:186+19] # 210 Observations when 5v5 and controlling 4 players
            elif self.n_agents == 3:
                encoded_obs = each_obs[:, 19:162+19] # 162 Observations when 4v4 and controlling 3 players

            # Extract the available actions for each agent
            avail_actions = torch.zeros((each_obs.shape[0], 20), device=each_obs.device)
            avail_actions[:, :19] = each_obs[:, :19]

            rnn_hidden_state = rnn_states[i].unsqueeze(1)

            # NOTE: Maybe should pass in the active_masks as well (EpisodeKey.DONES maybe)?
            if is_training:
                actions, rnn_hidden_state, action_log_probs, dist_entropy, action_log_probs_all = self.model(encoded_obs, rnn_hidden_state, available_actions=avail_actions, 
                                                                                       deterministic=not explore, masks=rnn_mask, actions=current_actions, active_masks=None)
            else:
                with torch.no_grad():
                    actions, rnn_hidden_state, action_log_probs, dist_entropy, action_log_probs_all = self.model(encoded_obs, rnn_hidden_state, available_actions=avail_actions, 
                                                                                           deterministic=not explore, masks=rnn_mask, actions=current_actions, active_masks=None)
            #print("available actions in actor: ", avail_actions) 
            #print("action log probs in actor: ", action_log_probs_all)

            # Check whether we want to dribble and we are sprinting (i.e. sprint is set to 0 in available_actions for this obs),
            # if so, change action to release sprinting instead of dribbling
            # NOTE: Is this already taken care of by passing available actions to the model iteslf?
            # NOTE: Maybe should assign a near-zero probability to action_log_probabilities as well...
            for step in range(actions.shape[0]):
                if actions[step] == DRIBBLE and each_obs[step, SPRINT] == 0:
                    actions[step] = RELEASE_SPRINT

            # TODO: Check the -1 index here. It should be fine, since in rollouts there is only one step and one hidden state per agent
            # Whle during training, there are multiple steps and hidden states per agent, so we need to keep track of the last one 
            self.rnn_hidden_state[i] = rnn_hidden_state[-1].detach()
            rnn_hidden_states[i] = rnn_hidden_state[-1].detach()

            action_list.append(actions)
            action_lps.append(action_log_probs)
            action_lps_all.append(action_log_probs_all)
            dist_entropies.append(dist_entropy)

        action_list = torch.cat(action_list).squeeze(-1)
        action_list = action_list.int()
        action_lps = torch.stack(action_lps).squeeze(-1)
        action_lps_all = torch.stack(action_lps_all)
        if None not in dist_entropies:
            dist_entropies = torch.cat(dist_entropies)
        else:
            dist_entropies = torch.ones_like(action_list)

        return action_list, rnn_hidden_states, action_lps, dist_entropies, action_lps_all
    
    def to_device(self, device):
        self.to(device)
        for i in range(self.n_agents):
            self.rnn_hidden_state[i] = self.rnn_hidden_state[i].to(device)
        self.model.to_device(device)
        return self 

class Critic(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        rnn_shape = [1, 512]
        self.rnn_layer_num = 1
        self.rnn_state_size = rnn_shape[-1]
        self.n_agents = int(custom_config["num_players_controlled"])
        self.model = ValueNetwork(self.n_agents)
        self.rnn_hidden_state = torch.zeros((1, rnn_shape[-1]), dtype=torch.float32)

    def forward(self, observation, rnn_states, rnn_masks, is_training=True):  
        observation = check_tensor(observation, device='cuda')  
        rnn_states = check_tensor(rnn_states, device='cuda')  
        rnn_masks = check_tensor(rnn_masks, device='cuda')  

        # Extract the global and local observations for all elements in the batch
        if self.n_agents == 10:
            global_obs = observation[:, 330+19:]
        elif self.n_agents == 5:
            global_obs = observation[:, 210+19:]
        elif self.n_agents == 4:
            global_obs = observation[:, 186+19:]
        elif self.n_agents == 3:
            global_obs = observation[:, 162+19:]

        # Remove duplicates from observations, rnn_states, and rnn_masks
        global_obs = global_obs.view(-1, self.n_agents, global_obs.shape[-1])[:, 0, :]
        rnn_masks = rnn_masks.view(-1, self.n_agents, rnn_masks.shape[-1])[:, 0, :]

        if is_training:  
            global_values, rnn_hidden_states = self.model(global_obs, rnn_states, rnn_masks)
        else:  
            with torch.no_grad():
                global_values, rnn_hidden_states = self.model(global_obs, rnn_states, rnn_masks)

        # Detach the hidden states from the computation graph as its gradients are not needed
        self.rnn_hidden_state = rnn_hidden_states.detach()
    
        return global_values, rnn_hidden_states

    def to_device(self, device):
        self.to(device)
        self.rnn_hidden_state = self.rnn_hidden_state.to(device)
        self.model.to_device(device)
        return self

# TODO(jh): we need a dummy one
class FeatureEncoder:
    def __init__(self, *args, **kwargs):
        self.n_players_per_team = kwargs["num_players"]
        pass

    def encode(self, states):
        # at least 19 for action masks
        raw_obs = []
        raw_shared_obs = []

        for i in states:
            current_obs = i.obs_list[-1]
            tartrl_obs = tartrl_obs_deal(current_obs, self.n_players_per_team // 2)
            obs = tartrl_obs['obs']
            share_obs = tartrl_obs['share_obs']
            avail_actions = tartrl_obs['available_action']

            obs_new = np.concatenate([avail_actions, obs])

            raw_obs.append(obs_new)
            raw_shared_obs.append(share_obs)

        # NOTE: Currently, all the share_obs are the same for each of the n_agents observations and np.concatenate repeats them in each one
        # Maybe change it so it only repeats them once?
            
        raw_obs_stacked = np.stack(raw_obs)     
        raw_shared_obs_stacked = np.stack(raw_shared_obs)

        # Concatenate raw_obs_stacked and raw_shared_obs_stacked along a new axis  
        encoded_obs = np.concatenate([raw_obs_stacked, raw_shared_obs_stacked], axis=1)

        return encoded_obs

    @property
    def global_observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=[
                20,
            ],
        )
    @property
    def action_space(self):
        return Discrete(19)


    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=[
                20,
            ],
        )

