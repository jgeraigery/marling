# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import copy
from distutils.log import Log
import os
import pickle
import random
import gym
import torch
import numpy as np

from torch import nn
from light_malib.utils.logger import Logger
from light_malib.utils.typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from light_malib.utils.episode import EpisodeKey

from light_malib.algorithm.common.policy import Policy

from ..utils import PopArt, init_fc_weights
import wrapt
import tree
import importlib
from light_malib.utils.logger import Logger
from light_malib.registry import registry


def hard_update(target, source):
    """Copy network parameters from source to target.

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15

    :param torch.nn.Module target: Net to copy parameters to.
    :param torch.nn.Module source: Net whose parameters to copy
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


@wrapt.decorator
def shape_adjusting(wrapped, instance, args, kwargs):
    """
    A wrapper that adjust the inputs to corrent shape.
    e.g.
        given inputs with shape (n_rollout_threads, n_agent, ...)
        reshape it to (n_rollout_threads * n_agent, ...)
    """
    offset = len(instance.preprocessor.shape)
    original_shape_pre = kwargs[EpisodeKey.CUR_OBS].shape[:-offset]
    num_shape_ahead = len(original_shape_pre)

    def adjust_fn(x):
        if isinstance(x, (np.ndarray,torch.Tensor)):
            return x.reshape((-1,) + x.shape[num_shape_ahead:])
        else:
            return x

    def recover_fn(x):
        if isinstance(x, (np.ndarray,torch.Tensor)):
            # If the tensor is reshapable
            if np.prod(x.shape) == np.prod(original_shape_pre + x.shape[1:]):
                return x.reshape(original_shape_pre + x.shape[1:])
            else:
                return x
        else:
            return x

    adjusted_args = tree.map_structure(adjust_fn, args)
    adjusted_kwargs = tree.map_structure(adjust_fn, kwargs)

    rets = wrapped(*adjusted_args, **adjusted_kwargs)

    recover_rets = tree.map_structure(recover_fn, rets)

    return recover_rets


@registry.registered(registry.POLICY)
class TiZero(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        self.random_exploration = False
        self.share_backbone = False

        # Make sure the model used is gr_football.tizero 
        assert model_config.get("model", "gr_football.tizero") == "gr_football.tizero", "This policy class only supports gr_football.tizero model" 
        model_type = model_config.get(
            "model", "gr_football.tizero"
        ) 

        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("light_malib.model.{}".format(model_type))

        FE_cfg = custom_config.get('FE_cfg', None)
        if FE_cfg is not None:
            self.feature_encoder = model.FeatureEncoder(**FE_cfg)
        else:
            self.feature_encoder = model.FeatureEncoder()

        # jh: re-define observation space based on feature encoder
        global_observation_space = self.feature_encoder.global_observation_space
        observation_space = self.feature_encoder.observation_space
        action_space = self.feature_encoder.action_space

        super(TiZero, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.num_agents = custom_config.get('num_players_controlled', 10)
        print(f'num_agents is {self.num_agents}')

        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )
        
        # TODO(jh): retrieve from feature encoder as well.
        # TODO(jh): this is not true in most cases, may be removed later.
        global_observation_space = observation_space

        self.actor = model.Actor(
            self.model_config["actor"],
            observation_space,
            action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        self.critic = model.Critic(
            self.model_config["critic"],
            global_observation_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        self.observation_space = observation_space
        self.action_space = action_space

        if custom_config["use_popart"]:
            self.value_normalizer = PopArt(
                1, device=self.device, beta=custom_config["popart_beta"]
            )

    def get_initial_state(self, batch_size) -> List[DataTransferType]:
        # TODO(jh): try to remove dependcies on rnn_layer_num & rnn_state_size
        return {
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(
                (batch_size, self.actor.rnn_layer_num, self.actor.rnn_state_size)
            ),
            # TODO: The critic should also follow using the batch size, but since it seems to always be 
            # set to the number of agents, it has been changed to 1 for now as the critic only outputs 
            # one value for all players. Should be changed to something that follows batch size.
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(
                (1, self.critic.rnn_layer_num, self.critic.rnn_state_size)
            ),
        }
    
    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy.actor = self_copy.actor.to_device(device)
        self_copy.critic = self_copy.critic.to_device(device)

        if self.custom_config["use_popart"]:
            self_copy.value_normalizer = self_copy.value_normalizer.to(device)
            self_copy.value_normalizer.tpdv = dict(dtype=torch.float32, device=device)
        return self_copy

    @shape_adjusting
    def compute_action(self, **kwargs):
        '''
        There are three ways of using this function.
        1. inference=True, explore=True, actions=None, used in rollouts for training. It will sample actions randomly.
        2. inference=True, explore=False, actions=None, used in rollouts for evaluation.It will use actions with max probs.
        3. inference=False, explore=False, actions=not None, used in training. It will evaluate log probs of actions.
        '''
        is_training = kwargs.get("is_training", True)
        inference=kwargs.get("inference",True)

        original_device = self.device
        if inference:
            self.device = 'cpu'
        
        # check if is numpy and convert to tensor
        # TODO(jh): numpy<->tensor conversion should be automatic. to_numpy should be automatically set.
        for k,v in kwargs.items():
            if isinstance(v,np.ndarray):
                v=torch.tensor(v, device=self.device,requires_grad=False)
                kwargs[k]=v

        actions=kwargs.get(EpisodeKey.ACTION,None)        
        explore=kwargs.get("explore",True)
        # when actions are provided, we simply evaluate at these actions.
        no_critic=kwargs.get("no_critic",False)
        to_numpy=kwargs.get("to_numpy",False)

        if not inference:
            explore=False
            
        with torch.set_grad_enabled(not inference):
            observations = kwargs[EpisodeKey.CUR_OBS]
            actor_rnn_states = kwargs[EpisodeKey.ACTOR_RNN_STATE]
            critic_rnn_states = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            action_masks = kwargs[EpisodeKey.ACTION_MASK]
            rnn_masks = kwargs[EpisodeKey.DONE]
            if EpisodeKey.CUR_STATE not in kwargs:
                states = observations
            else:
                states = kwargs[EpisodeKey.CUR_STATE]
            actions, actor_rnn_states, action_log_probs, dist_entropy, action_log_probs_all = self.actor(
                observations, actor_rnn_states, rnn_masks, action_masks, explore, actions, is_training = is_training
            )

            # print("action log probs all = ", action_log_probs_all)
            
            # TODO(jh): add to_numpy
            if to_numpy:
                actor_rnn_states = actor_rnn_states.detach().cpu().numpy()
                actions = actions.detach().cpu().numpy() 
                if self.random_exploration:
                    exploration_actions = np.zeros(actions.shape, dtype=int)
                    for i in range(len(actions)):
                        if random.uniform(0, 1) < self.random_exploration:
                            exploration_actions[i] = int(random.choice(range(19)))
                        else:
                            exploration_actions[i] = int(actions[i])
                    actions = exploration_actions

                action_log_probs = action_log_probs.detach().cpu().numpy() 
                action_log_probs_all = action_log_probs_all.detach().cpu().numpy()

            ret = {
                EpisodeKey.ACTION_LOG_PROB: action_log_probs,
                EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states,
                EpisodeKey.ACTION_LOG_PROB + "_all": action_log_probs_all.reshape(-1, action_log_probs_all.shape[-1]),
            }
            
            if kwargs.get(EpisodeKey.ACTION,None) is None:
                ret[EpisodeKey.ACTION]=actions
            else:
                ret[EpisodeKey.ACTION_ENTROPY]=dist_entropy

            # TODO: Maybe we should add "not inference" to not evaluate the critic during inference 
            if not no_critic: 
                values, critic_rnn_states = self.critic(
                    states, critic_rnn_states, rnn_masks
                )

                if to_numpy:
                    values = values.detach().cpu().numpy()
                    critic_rnn_states = critic_rnn_states.detach().cpu().numpy()

                ret[EpisodeKey.STATE_VALUE]=values
                ret[EpisodeKey.CRITIC_RNN_STATE]=critic_rnn_states

            self.device = original_device
            return ret

    @shape_adjusting
    def value_function(self, **kwargs):       

        inference=kwargs.get("inference",True)

        original_device = self.device
        if inference:
            self.device = 'cpu'

        # check if is numpy and convert to tensor
        for k,v in kwargs.items():
            if isinstance(v,np.ndarray):
                v=torch.tensor(v, device=self.device, requires_grad=False)
                kwargs[k]=v
                
        to_numpy=kwargs.get("to_numpy",False)
                    
        # only used in inference now        
        with torch.set_grad_enabled(not inference):
            observations = kwargs[EpisodeKey.CUR_OBS]
            if EpisodeKey.CUR_STATE not in kwargs:
                states = observations
            else:
                states = kwargs[EpisodeKey.CUR_STATE]
            critic_rnn_state = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            rnn_mask = kwargs[EpisodeKey.DONE]
                
            value, _ = self.critic(states, critic_rnn_state, rnn_mask)
            
            if to_numpy:
                value = value.cpu().numpy()
            
            self.device = original_device
            return {EpisodeKey.STATE_VALUE: value}

    def train(self):
        self.actor.train()
        self.critic.train()
        if self.custom_config["use_popart"]:
            self.value_normalizer.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        if self.custom_config["use_popart"]:
            self.value_normalizer.eval()

    def dump(self, dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(self.actor, os.path.join(dump_dir, "actor.pt"))
        torch.save(self.critic, os.path.join(dump_dir, "critic.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        res = TiZero(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )

        actor_path = os.path.join(dump_dir, "actor.pt")
        if os.path.exists(actor_path):
            actor = torch.load(actor_path, res.device)
            hard_update(res.actor, actor)
            
        critic_path = os.path.join(dump_dir, "critic.pt")
        if os.path.exists(critic_path):
            critic = torch.load(critic_path, res.device)
            hard_update(res.critic, critic)
        
        return res
