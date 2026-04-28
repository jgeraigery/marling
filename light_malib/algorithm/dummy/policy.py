# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import copy
import os
import pickle
import gym
import torch
import numpy as np

from light_malib.utils.episode import EpisodeKey

from light_malib.algorithm.common.policy import Policy

from light_malib.model.gr_football.tizero import FeatureEncoder

import wrapt
import tree
from light_malib.registry import registry
from typing import Dict, Any

from gym.spaces import Box, Discrete

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
            return x.reshape(original_shape_pre + x.shape[1:])
        else:
            return x

    adjusted_args = tree.map_structure(adjust_fn, args)
    adjusted_kwargs = tree.map_structure(adjust_fn, kwargs)

    rets = wrapped(*adjusted_args, **adjusted_kwargs)

    recover_rets = tree.map_structure(recover_fn, rets)

    return recover_rets


@registry.registered(registry.POLICY)
class Dummy(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space = None,
        action_space: gym.spaces.Space = None,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        
        custom_config = {}

        action_space = Discrete(19)
        observation_space = Box(low=-1, high=1, shape=[20])

        super(Dummy, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", True) else "cpu"
        )

        # Just a dummy feature encoder.
        FE_cfg = {
            "num_players": 22,
        }
        self.feature_encoder = FeatureEncoder(**FE_cfg)
       
        self.observation_space = observation_space
        self.action_space = action_space

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(
                (batch_size, 1, 1)
            ),
            # TODO: The critic should also follow using the batch size, but since it seems to always be 
            # set to the number of agents, it has been changed to 1 for now as the critic only outputs 
            # one value for all players. Should be changed to something that follows batch size.
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(
                (1, 1, 1)
            ),
        }

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        return self_copy

    @shape_adjusting
    def compute_action(self, **kwargs):
        '''
        NOTE(jh): there are three ways of using this function.
        1. inference=True, explore=True, actions=None, used in rollouts for training. It will sample actions randomly.
        2. inference=True, explore=False, actions=None, used in rollouts for evaluation.It will use actions with max probs.
        3. inference=False, explore=False, actions=not None, used in training. It will evaluate log probs of actions.
        '''
        observations = kwargs[EpisodeKey.CUR_OBS]
        if EpisodeKey.CUR_STATE not in kwargs:
            states = observations
        else:
            states = kwargs[EpisodeKey.CUR_STATE]
            
        # Return no_op action for all observations
        ret = {
            EpisodeKey.ACTION: np.zeros((observations.shape[0]), dtype=np.int16),
        }
            
        return ret
    
    @shape_adjusting
    def value_function(self, *args, **kwargs):
        return super().value_function(*args, **kwargs)
    
    def train(self):
        return super().train()
    
    def eval(self):
        return super().eval()

    def dump(self, dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))