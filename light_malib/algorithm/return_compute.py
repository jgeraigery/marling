# Modified portions of the file are Copyright (c) 2026 Electronic Arts Inc.

# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from math import inf
from cv2 import norm
from matplotlib.pylab import f
import torch
import numpy as np

from light_malib.reward_shaping.intrinsic_bonus.random_network_distillation import rnd
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger


def compute_return(policy, batch, intrinsic_reward_network=None, rnd_bonus=None, obs_normalizer=None, exploration_bonus_normalizer=None,
                   intrinsic_reward_normalizer=None):

    return_mode = policy.custom_config["return_mode"]
    if return_mode == "gae":
        return compute_new_gae(policy,batch,use_old_V=True)
    elif return_mode == "vtrace":
        raise NotImplementedError
    elif return_mode in ["new_gae", "async_gae"]:
        return compute_new_gae(policy, batch, intrinsic_reward_network=intrinsic_reward_network, rnd_bonus=rnd_bonus,
                                 obs_normalizer=obs_normalizer, exploration_bonus_normalizer=exploration_bonus_normalizer,
                                 intrinsic_reward_normalizer=intrinsic_reward_normalizer)
    elif return_mode in ["new_gae_trace"]:
        return compute_new_gae_trace(policy, batch)
    elif return_mode in ["mc"]:
        return compute_mc(policy, batch, use_old_V=True)
    elif return_mode in ["new_mc"]:
        return compute_mc(policy, batch)
    else:
        raise ValueError("Unexpected return mode: {}".format(return_mode))


def compute_new_gae(policy, batch, use_old_V=False, intrinsic_reward_network=None,
                    rnd_bonus=None, obs_normalizer=None, exploration_bonus_normalizer=None,
                    intrinsic_reward_normalizer=None):
    """
    NOTE the last obs,state,done,critic_rnn_states are for bootstraping.
    """
    # TODO: batch['done'] is not correctly set here I think
    # TODO: Should there be a torch.no_grad() block here?
    with torch.no_grad():
        cfg = policy.custom_config
        gamma, gae_lambda = cfg["gamma"], cfg["gae"]["gae_lambda"]
        rewards = batch[EpisodeKey.REWARD]
        actions = batch[EpisodeKey.ACTION]
        dones = batch[EpisodeKey.DONE]
        cur_states = batch[EpisodeKey.CUR_STATE]
        cur_obs = batch[EpisodeKey.CUR_OBS]
        rnn_states = batch[EpisodeKey.CRITIC_RNN_STATE]

        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert (
            rewards.shape[1] == Tp1 - 1
            and dones.shape[1] == Tp1
            and rnn_states.shape[1] == Tp1
            and cur_states.shape[1] == Tp1
        ), "{}".format({k: v.shape for k, v in batch.items()})

        if not use_old_V:
            policy.eval()
            ret = policy.value_function(
                **{
                    EpisodeKey.CUR_STATE: cur_states,
                    EpisodeKey.CUR_OBS: cur_obs,
                    EpisodeKey.CRITIC_RNN_STATE: rnn_states,
                    EpisodeKey.DONE: dones
                },
                inference=True
            )
            normalized_value=ret[EpisodeKey.STATE_VALUE]
        else:
            normalized_value=ret[EpisodeKey.STATE_VALUE]

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(
                normalized_value.reshape(-1, normalized_value.shape[-1])
            )
            values = values.reshape(normalized_value.shape)
        else:
            # Denormalize the values using critic's running stats
            normalized_value = policy.critic.module.model.running_stats.denormalize(normalized_value)
            values = normalized_value

        # TODO: Probably remove this assert statement after testing and debugging. It is just slowing it down.
        # Find out if the values in fourth dimension are the same for all third dimensions in any time step
        assert False not in [torch.all(rewards[:, :, i] == rewards[:, :, i+1]) for i in range(N - 1)] and \
               False not in [torch.all(dones[:, :, i] == dones[:, :, i+1]) for i in range(N - 1)], \
               "The values in the fourth dimension are not the same for all third dimensions in any time step"

        # Remove duplicate reward and done values for each agent
        rewards = rewards[:, :, 0].squeeze(-1)
        dones = dones[:, :, 0].squeeze(-1)

        # View values in the correct form of [B, Tp1] instead of [B, Tp1, 1]
        values = values.view(B, Tp1)

        # Ensure values, dones, and rewards are on the same device as the rewards
        values = values.to(rewards.device)
        dones = dones.to(rewards.device)
        
        gae = 0
        gae_intrinsic = 0
        adv_shape = (B, Tp1 - 1) # Only one value per time-step, as the advantage value is global
        advantages = torch.zeros(adv_shape, device=rewards.device)
        delta_list = torch.zeros(adv_shape, device=rewards.device)
        advantages_with_intrinsic = torch.zeros(adv_shape, device=rewards.device)
        delta_list_intrinsic = torch.zeros(adv_shape, device=rewards.device)
        intrinsic_rewards = torch.zeros(adv_shape, device=rewards.device)
        intrinsic_exploration = torch.zeros(adv_shape, device=rewards.device)

        # Initialize the hidden state of the GRU
        if intrinsic_reward_network is not None:
            intrinsic_reward_network.reset_hidden(B)

        if rnd_bonus is not None:
            Logger.info("Using RND bonus to compute returns.")
        if intrinsic_reward_network is not None:
            Logger.info("Using SSIR to compute returns.")

        for t in reversed(range(Tp1 - 1)):

            # TODO: we should differentiate terminal case and truncation case. now we directly follow env's dones.
            # More specifically, the env's dones don't indicate the truncation case.
            delta = (
                rewards[:, t] 
                + gamma * (1 - dones[:, t]) * values[:, t + 1]
                - values[:, t]
            )
            gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * gae
            # gae *= (1-done[t])          #terminal case
            advantages[:, t] = gae
            delta_list[:, t] = delta

            # Calculate the intrinsic reward and intrinsic bonus and add that to the original reward
            if intrinsic_reward_network or rnd_bonus is not None:

                # Set the offset for the start of the global observation
                if N == 10:
                    obs_global_start_offset = 330+19
                elif N == 5:
                    obs_global_start_offset = 210+19
               
                # Calculate the intrinsic reward
                if intrinsic_reward_network is not None:

                    # Use the private observations of each agent to calculate the intrinsic reward
                    obs_batch_intrinsic = cur_obs[:, t]
                    obs_batch_intrinsic = obs_batch_intrinsic[:, :, 19:obs_global_start_offset] 
                
                    intrinsic_reward = intrinsic_reward_network.get_intrinsic_reward(obs_batch_intrinsic, actions[:, t].to(torch.int32))

                    # Average the intrinsic reward over the number of agents
                    intrinsic_reward = intrinsic_reward.mean(dim=1)

                    # Update our intrinsic reward normalizer
                    # intrinsic_reward_normalizer.update(intrinsic_reward)
                    # Normalize the intrinsic reward
                    # intrinsic_reward = intrinsic_reward_normalizer.normalize_reward(intrinsic_reward)

                    # Weight the intrinsic reward by a coefficient
                    intrinsic_reward = intrinsic_reward
                    intrinsic_rewards[:, t] = intrinsic_reward

                # Calculate the intrinsic exploration bonus
                if rnd_bonus is not None:

                    # Take the observation for only one player because they are global observation and the same for all players
                    obs_batch_intrinsic = cur_obs[:, t, 0, obs_global_start_offset:] 

                    # Update our observation normalizer
                    obs_normalizer.update(obs_batch_intrinsic)

                    # Normalize our observations
                    obs_normalized = obs_normalizer.normalize_obs(obs_batch_intrinsic)

                    # Calculate the intrinsic RND bonus
                    intrinsic_bonus = rnd_bonus.compute_intrinsic_reward(obs_normalized)
                    intrinsic_bonus = intrinsic_bonus.unsqueeze(-1)

                    # Update our intrinsic exploration bonus reward noramlizer
                    exploration_bonus_normalizer.update(intrinsic_bonus)
                    
                    # Normalize the intrinsic bonus
                    intrinsic_bonus = exploration_bonus_normalizer.normalize_reward(intrinsic_bonus)

                    # Weight the intrinisc bonus by a given coefficient
                    intrinsic_bonus = intrinsic_bonus * policy.custom_config["exploration_bonus_coefficient"]
                    intrinsic_bonus = intrinsic_bonus.squeeze(-1)

                    # Store the intrinsic bonus in the intrinsic exploration tensor
                    intrinsic_exploration[:, t] = intrinsic_bonus

                if intrinsic_reward_network is not None and rnd_bonus is not None:
                    delta_intrinsic = (
                    (rewards[:, t] + intrinsic_reward + intrinsic_bonus)
                    + gamma * (1 - dones[:, t]) * values[:, t + 1]
                    - values[:, t]
                    )

                elif intrinsic_reward_network is not None:
                    delta_intrinsic = (
                    (rewards[:, t] + intrinsic_reward)
                    + gamma * (1 - dones[:, t]) * values[:, t + 1]
                    - values[:, t]
                    )
                
                elif rnd_bonus is not None:
                    delta_intrinsic = (
                    (rewards[:, t] + intrinsic_bonus)
                    + gamma * (1 - dones[:, t]) * values[:, t + 1]
                    - values[:, t]
                    )
                
                gae_intrinsic = delta_intrinsic + gamma * gae_lambda * (1 - dones[:, t]) * gae_intrinsic
                advantages_with_intrinsic[:, t] = gae_intrinsic
                delta_list_intrinsic[:, t] = delta_intrinsic

        returns = advantages + values[:, :-1]
        if intrinsic_reward_network or rnd_bonus is not None:
            returns_with_intrinsic = advantages_with_intrinsic + values[:, :-1]

        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(
                returns.reshape(-1, rewards.shape[-1])
            )
            normalized_returns = normalized_returns.reshape(rewards.shape)

            if intrinsic_reward_network or rnd_bonus is not None:
                normalized_returns_with_intrinsic = policy.value_normalizer(
                    returns_with_intrinsic.reshape(-1, rewards.shape[-1])
                )
                normalized_returns_with_intrinsic = normalized_returns_with_intrinsic.reshape(rewards.shape)
        else:
            normalized_returns = returns
            if intrinsic_reward_network or rnd_bonus is not None:
                normalized_returns_with_intrinsic = returns_with_intrinsic

        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())
        advantages_with_intrinsic = (advantages_with_intrinsic - advantages_with_intrinsic.mean()) / (1e-9 + advantages_with_intrinsic.std())

        ret = {
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: values[:, :-1],
            EpisodeKey.ADVANTAGE: advantages,
            "delta": delta_list,
        }

        if intrinsic_reward_network or rnd_bonus is not None:
            ret[EpisodeKey.RETURN + "_with_intrinsic"] = normalized_returns_with_intrinsic
            ret[EpisodeKey.ADVANTAGE + "_with_intrinsic"] = advantages_with_intrinsic
            ret["delta_with_intrinsic"] = delta_list_intrinsic
            if intrinsic_reward_network is not None:
                ret["intrinsic_reward"] = intrinsic_rewards
            if rnd_bonus is not None:
                ret["intrinsic_exploration"] = intrinsic_exploration

        for key in batch:
            if key in [
                EpisodeKey.CUR_OBS,
                EpisodeKey.DONE,
                EpisodeKey.CRITIC_RNN_STATE,
                EpisodeKey.CUR_STATE,
            ]:
                # remove bootstraping data
                ret[key] = batch[key][:, :-1]
            else:
                ret[key] = batch[key]

        # Ensure that the shape of the value estimates in "batch", passed in arguments is the same as the shape of the value estimates computed here
        ret[EpisodeKey.STATE_VALUE] = ret[EpisodeKey.STATE_VALUE].view(values[:, :-1].shape)

        return ret
    
def compute_new_gae_trace(policy, batch):
    with torch.no_grad():
        cfg = policy.custom_config
        gamma, gae_lambda = cfg["gamma"], cfg["gae"]["gae_lambda"]
        rewards = batch[EpisodeKey.REWARD]
        dones = batch[EpisodeKey.DONE]
        cur_states = batch[EpisodeKey.CUR_STATE]
        cur_obs = batch[EpisodeKey.CUR_OBS]
        actor_rnn_states = batch[EpisodeKey.ACTOR_RNN_STATE]
        critic_rnn_states = batch[EpisodeKey.CRITIC_RNN_STATE]
        actions = batch[EpisodeKey.ACTION]
        action_masks = batch[EpisodeKey.ACTION_MASK]
        old_action_log_probs = batch[EpisodeKey.ACTION_LOG_PROB]

        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert (
            rewards.shape[1] == Tp1 - 1
            and dones.shape[1] == Tp1
            and critic_rnn_states.shape[1] == Tp1
            and cur_states.shape[1] == Tp1
        ), "{}".format({k: v.shape for k, v in batch.items()})

        policy.eval()
        ret = policy.compute_action(
            **{
                EpisodeKey.CUR_STATE: cur_states[:,:-1],
                EpisodeKey.CUR_OBS: cur_obs[:,:-1],
                EpisodeKey.ACTION: actions,
                EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states[:,:-1],
                EpisodeKey.DONE: dones[:,:-1],
                EpisodeKey.ACTION_MASK: action_masks 
            },
            inference=True,
            explore=False,
            no_critic=True
        )
        action_log_probs=ret[EpisodeKey.ACTION_LOG_PROB]
        
        # batch_size,episode_length,num_agents
        imp_weights = torch.exp(
            action_log_probs - old_action_log_probs
        )
        
        # batch_size,episode_length,1,num_agents
        imp_weights = imp_weights.unsqueeze(dim=2)
        
        # batch_size,episode_length,1,num_agents
        imp_weights = imp_weights.repeat_interleave(N, dim=2)

        # batch_size,episode_length,1,1
        imp_weights = imp_weights.prod(dim=3,keepdim=True)
        imp_weights = imp_weights.clamp(max=1.0)
        
        ret = policy.value_function(
            **{
                EpisodeKey.CUR_STATE: cur_states,
                EpisodeKey.CUR_OBS: cur_obs,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states,
                EpisodeKey.DONE: dones
            },
            inference=True
        )
        normalized_value=ret[EpisodeKey.STATE_VALUE]

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(
                normalized_value.reshape(-1, normalized_value.shape[-1])
            )
            values = values.reshape(normalized_value.shape)
        else:
            values = normalized_value

        gae = 0
        advantages = torch.zeros_like(rewards)
        delta_list = torch.zeros_like(rewards)

        for t in reversed(range(Tp1 - 1)):
            delta = (
                rewards[:, t]
                + gamma * (1 - dones[:, t]) * values[:, t + 1]
                - values[:, t]
            )
            # NOTE(jh): imp_weights is at t+1 here, the immediate reward has no need to be Importance .
            gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * gae * (imp_weights[:, t+1] if t<Tp1-2 else 1)
            # TODO(jh): we should differentiate terminal case and truncation case. now we directly follow env's dones
            # gae *= (1-done[t])          #terminal case
            advantages[:, t] = gae
            delta_list[:, t] = delta

        # TODO(jh): do we need * imp_weights here?
        returns = advantages + values[:, :-1]

        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(
                returns.reshape(-1, rewards.shape[-1])
            )
            normalized_returns = normalized_returns.reshape(rewards.shape)
        else:
            normalized_returns = returns

        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())

        ret = {
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: normalized_value[:, :-1],
            EpisodeKey.ADVANTAGE: advantages,
            "delta": delta_list,
        }

        for key in batch:
            if key in [
                EpisodeKey.CUR_OBS,
                EpisodeKey.DONE,
                EpisodeKey.CRITIC_RNN_STATE,
                EpisodeKey.CUR_STATE,
            ]:
                # remove bootstraping data
                ret[key] = batch[key][:, :-1]
            else:
                ret[key] = batch[key]

        return ret


def compute_mc(policy, batch, use_old_V=False):
    with torch.no_grad():
        cfg = policy.custom_config
        gamma = cfg["gamma"]
        rewards = batch[EpisodeKey.REWARD]
        dones = batch[EpisodeKey.DONE]
        cur_states = batch[EpisodeKey.CUR_STATE]
        cur_obs = batch[EpisodeKey.CUR_OBS]
        rnn_states = batch[EpisodeKey.CRITIC_RNN_STATE]

        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert (
            rewards.shape[1] == Tp1 - 1
            and dones.shape[1] == Tp1
            and rnn_states.shape[1] == Tp1
            and cur_states.shape[1] == Tp1
        ), "{}".format({k: v.shape for k, v in batch.items()})

        if not use_old_V:
            policy.eval()
            ret = policy.value_function(
                **{
                    EpisodeKey.CUR_STATE: cur_states,
                    EpisodeKey.CUR_OBS: cur_obs,
                    EpisodeKey.CRITIC_RNN_STATE: rnn_states,
                    EpisodeKey.DONE: dones
                },
                inference=True
            )
            normalized_value=ret[EpisodeKey.STATE_VALUE]
        else:
            normalized_value=batch[EpisodeKey.STATE_VALUE]

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(
                normalized_value.reshape(-1, normalized_value.shape[-1])
            )
            values = values.reshape(normalized_value.shape)
        else:
            values = normalized_value

        ret = 0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(Tp1 - 1)):
            ret = gamma * (1 - dones[:, t]) * ret + rewards[:, t]
            if t == Tp1 - 1 - 1:
                # bootstrapping values
                ret += (1 - dones[:, t]) * values[:, t + 1]
            advantages[:, t] = ret - values[:, t]

        returns = advantages + values[:, :-1]

        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(
                returns.reshape(-1, rewards.shape[-1])
            )
            normalized_returns = normalized_returns.reshape(rewards.shape)
        else:
            normalized_returns = returns

        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())

        ret = {
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: normalized_value[:, :-1],
            EpisodeKey.ADVANTAGE: advantages
        }

        for key in batch:
            if key in [
                EpisodeKey.CUR_OBS,
                EpisodeKey.DONE,
                EpisodeKey.CRITIC_RNN_STATE,
                EpisodeKey.CUR_STATE,
            ]:
                # remove bootstraping data
                ret[key] = batch[key][:, :-1]
            else:
                ret[key] = batch[key]

        return ret
