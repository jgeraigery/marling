# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from tkinter import W
from typing import Union
from matplotlib.pylab import f
import torch
from light_malib.model.gr_football import basic
from light_malib.reward_shaping import intrinsic_reward
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.logger import Logger
from light_malib.registry import registry
from light_malib.utils.running_stats import RunningStats
import numpy as np

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return (e**2) / 2


def to_value(tensor: torch.Tensor):
    return tensor.detach().cpu().item()


def basic_stats(name, tensor: torch.Tensor):
    stats = {}
    stats["{}_max".format(name)] = to_value(tensor.max())
    stats["{}_min".format(name)] = to_value(tensor.min())
    stats["{}_mean".format(name)] = to_value(tensor.mean())
    stats["{}_std".format(name)] = to_value(tensor.std())
    return stats


@registry.registered(registry.LOSS)
class TiZeroLoss(LossFunc):
    def __init__(self, intrinsic_reward_network=None, rnd_bonus = None, obs_normalizer=None, use_reward_shaping_after=5000):
        super(TiZeroLoss, self).__init__()

        self._use_clipped_value_loss = True
        self._use_huber_loss = False
        if self._use_huber_loss:
            self.huber_delta = 10.0
        self._use_max_grad_norm = True

        # Intrinsic reward network
        self.intrinsic_reward_network = intrinsic_reward_network
        # Observation normalizer for RND intrinsic bonus
        self.obs_normalizer = obs_normalizer
        # RND Intrinsic Bonus
        self.rnd_bonus = rnd_bonus
        # Use reward shaping after a certain number of epochs
        self.use_reward_shaping_after = use_reward_shaping_after

        # Running stats for the returns
        self.return_running_stats = RunningStats()


    def reset(self, policy, config):
        """
        reset should always be called for each training task.
        """
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()
        
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

        self.sub_algorithm_name = policy.custom_config.get("sub_algorithm_name","TiZero")   
        assert self.sub_algorithm_name in ["TiZero", "MAPPO","CoPPO","HAPPO","A2PO"]
        
        if self.sub_algorithm_name in ["TiZero", "IPPO", "MAPPO"]:
            self._use_seq=False
            self._use_two_stage=False
            self._use_co_ma_ratio=False
            self._clip_before_prod=False
            self._clip_others=False
            self._num_agents = policy.num_agents

        elif self.sub_algorithm_name=="CoPPO":
            self._use_seq=False
            self._use_two_stage=False
            self._use_co_ma_ratio=True
            self._clip_before_prod=True
            self._clip_others=True
            self._other_clip_param=policy.custom_config["other_clip_param"]
            self._num_agents=policy.custom_config["num_agents"]

        elif self.sub_algorithm_name=="HAPPO":
            self._use_seq=True
            self._use_two_stage=False
            self._use_co_ma_ratio=True
            self._clip_before_prod=True
            self._clip_others=False
            self._num_agents=policy.custom_config["num_agents"]
            self._seq_strategy=policy.custom_config.get("seq_strategy","random")
            # TODO(jh): check default
            self._one_agent_per_update=False
            self._use_agent_block=policy.custom_config.get("use_agent_block",False)
            if self._use_agent_block:
                self._block_num=policy.custom_config["block_num"]
            self._use_cum_sequence=True
            self._agent_seq=[]

        elif self.sub_algorithm_name=="A2PO":
            self._use_seq=True
            self._use_two_stage=True
            self._use_co_ma_ratio=True
            self._clip_before_prod=False
            self._clip_others=True
            self._other_clip_param=policy.custom_config["other_clip_param"]
            self._num_agents=policy.custom_config["num_agents"]
            self._seq_strategy=policy.custom_config.get("seq_strategy","semi_greedy")
            # TODO(jh): check default
            self._one_agent_per_update=False
            self._use_agent_block=policy.custom_config.get("use_agent_block",False)
            if self._use_agent_block:
                self._block_num=policy.custom_config["block_num"]
            self._use_cum_sequence=True
            self._agent_seq=[]

        else:
            raise NotImplementedError     
            
    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""
        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
        
        # TODO(jh): update actor and critic simutaneously
        param_groups=[]

        if len(list(self._policy.actor.parameters()))>0:
            param_groups.append({'name': 'actor', 'params': self.policy.actor.parameters(), 'lr': self._params["actor_lr"]})
        
        if len(list(self._policy.critic.parameters()))>0:
            param_groups.append({'name': 'critic', 'params': self.policy.critic.parameters(), 'lr': self._params["critic_lr"]})
        
        self.optimizer=optim_cls(
            param_groups,
            eps=self._params["opti_eps"],
            weight_decay=self._params["weight_decay"]
        )
        
        self.optimizer.zero_grad()
        
        self.n_opt_steps=0
        self.grad_accum_step=self._params.get("grad_accum_step",1)
        
    def loss_compute(self, sample, current_epoch):
        self.n_opt_steps+=1
        
        policy = self._policy
        policy.train()                
        return self.loss_compute_simultaneous(sample, current_epoch)
            
    def _select_data_from_agent_ids(
        self,
        x: Union[np.ndarray, torch.Tensor],
        agent_ids: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        '''
        we assume x is the shape [#batch_size*#agents,...]
        '''
        # TODO: Remove this assertion after testing, as it would probably not be needed.
        assert agent_ids is None, f'Agent ids is not none, it is {agent_ids}'
        if agent_ids is None:
            return x        
        
        if not isinstance(x,(np.ndarray,torch.Tensor)):
            return x
        
        x = x.reshape(-1, self._num_agents, *x.shape[1:])[:, agent_ids]
        x = x.reshape(-1,*x.shape[2:])
        return x

    def loss_compute_simultaneous(
        self, 
        sample,
        current_epoch,
        agent_ids:np.ndarray=None,
        update_actor:bool=True
    ):
        # agent_ids not None means block update
        if agent_ids is not None:
            assert len(agent_ids.shape)==1
        
        (
            share_obs_batch,
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            adv_targ,
            delta,
        ) = (
            sample[EpisodeKey.CUR_STATE],
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.RETURN],
            sample.get(EpisodeKey.ACTIVE_MASK, None),
            sample[EpisodeKey.ACTION_LOG_PROB],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTOR_RNN_STATE],
            sample[EpisodeKey.CRITIC_RNN_STATE],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.ADVANTAGE],
            sample["delta"],
        )

        # If reward shaping has been used, then the return, advantage, and delta computed using those is also included in the sample
        # And so, we use those values instead of the ones computed without intrinsic reward to optimize our policy
        # Also, we use the reward shaped return if the number of optimization steps is greater than a given number of steps, to mitigate initialization issues
        if EpisodeKey.RETURN + "_with_intrinsic" in sample.keys() and current_epoch > self.use_reward_shaping_after:
            Logger.info(f'Using shaped reward ({"SSIR + " if self.intrinsic_reward_network is not None else ""}{"RND" if self.rnd_bonus is not None else ""}) for optimization at training epoch {current_epoch}')
            return_batch = sample[EpisodeKey.RETURN + "_with_intrinsic"]
            adv_targ = sample[EpisodeKey.ADVANTAGE + "_with_intrinsic"]
            delta = sample["delta_with_intrinsic"]

        if update_actor:
            ret = self._policy.compute_action(
                **{
                    EpisodeKey.CUR_STATE: share_obs_batch,
                    EpisodeKey.CUR_OBS: obs_batch,
                    EpisodeKey.ACTION: actions_batch,
                    EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states_batch,
                    EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states_batch,
                    EpisodeKey.DONE: dones_batch,
                    EpisodeKey.ACTION_MASK: available_actions_batch  
                },
                inference=False,
                explore=False,
                is_training=True
            )


            values=ret[EpisodeKey.STATE_VALUE]
            action_log_probs=ret[EpisodeKey.ACTION_LOG_PROB]
            dist_entropy=ret[EpisodeKey.ACTION_ENTROPY]     
            action_log_probs_all=ret[EpisodeKey.ACTION_LOG_PROB + "_all"]

             # ============================== Policy Loss ================================

            # Here we calculate the Joint-Policy Ratio Optimization (JRPO) loss if we are using TiZero
            if self.sub_algorithm_name == "TiZero": 
                joint_log_probs = action_log_probs.permute(1, 0).sum(dim=1)  
                joint_old_log_probs = old_action_log_probs_batch.view(-1, self._num_agents).sum(dim=1)  
                imp_weights = torch.exp(joint_log_probs - joint_old_log_probs).view(-1, 1)
                approx_kl = (joint_old_log_probs - joint_log_probs).mean().item()
            else:  
                imp_weights = torch.exp(  
                    action_log_probs - old_action_log_probs_batch  
                ).view(-1,1)  
                approx_kl = (  
                    (old_action_log_probs_batch - action_log_probs).mean().item()  
                )  

            imp_weights = self._select_data_from_agent_ids(imp_weights, agent_ids)
            adv_targ = self._select_data_from_agent_ids(adv_targ, agent_ids)
            active_masks_batch = self._select_data_from_agent_ids(active_masks_batch,agent_ids)
            dist_entropy = self._select_data_from_agent_ids(dist_entropy, agent_ids)

            adv_targ = adv_targ.unsqueeze(-1)

            surr1 = imp_weights * adv_targ
            surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
            )
            
            if active_masks_batch is not None:
                surr = torch.min(surr1, surr2)
                policy_action_loss = (
                    -torch.sum(surr, dim=-1, keepdim=True) * active_masks_batch
                ).sum() / (active_masks_batch.sum()+1e-20)
                assert dist_entropy.shape==active_masks_batch.shape
                policy_entropy_loss = - (dist_entropy*active_masks_batch).sum()/(active_masks_batch.sum()+1e-20)
            else:
                surr = torch.min(surr1, surr2)
                policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()
                policy_entropy_loss = -dist_entropy.mean()

            # Adaptive policy entropy adjustment coefficient, given in https://arxiv.org/pdf/2405.04664v1
            # tau = 300
            # self.return_running_stats.update(return_batch)
            # rb = return_batch.reshape(-1, tau)
            # rb_mean = torch.mean(rb, dim = 0)
            # adaptive_entropy_coef = (1.0 / self.return_running_stats.max) * (rb_mean[-tau:].sum() / tau)

            policy_loss = policy_action_loss + policy_entropy_loss * self._policy.custom_config["entropy_coef"]

        else:
            ret = self._policy.value_function(
                **{
                    EpisodeKey.CUR_STATE: share_obs_batch,
                    EpisodeKey.CUR_OBS: obs_batch,
                    EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states_batch,
                    EpisodeKey.DONE: dones_batch
                },
                inference=False
            )
            values=ret[EpisodeKey.STATE_VALUE]
            
            policy_loss = 0
            active_masks_batch = self._select_data_from_agent_ids(active_masks_batch, agent_ids)

        # ============================== Intrinsic Reward Update ================================

        # If the intrinsic reward network is used, then we update the intrinsic reward network if the update frequency is met
        if self.intrinsic_reward_network is not None:
            # If we are in the final ppo epoch for the current batch and
            # we are at the update frequency for the intrinsic reward network, then we update the intrinsic reward network
            if self.n_opt_steps % self._policy.custom_config["ppo_epoch"] == 0 and \
               current_epoch % self._policy.custom_config.get("intrinsic_reward_update_freq", 2) == 0:
                Logger.info("Updating intrinsic reward network")
                self.intrinsic_reward_network.update_intrinsic_reward(sample, action_log_probs_all, self._num_agents)
        
        # ============================== Value Loss ================================
        values = self._select_data_from_agent_ids(values, agent_ids)
        value_preds_batch = self._select_data_from_agent_ids(value_preds_batch, agent_ids)
        orig_return_batch = self._select_data_from_agent_ids(sample[EpisodeKey.RETURN], agent_ids)

        values = values.view(value_preds_batch.shape)

        assert values.shape == value_preds_batch.shape, "Shape mismatch between values and value_preds_batch"
 
        # Update the running stats of our critic on the new rewards if it is the final ppo epoch for the current batch
        if self.n_opt_steps % self._policy.custom_config["ppo_epoch"] == 0:
            # Update the running stats of our critic on the new rewards
            self._policy.critic.module.model.running_stats.update(orig_return_batch)
 
        value_loss = self._calc_value_loss(
            values, value_preds_batch, orig_return_batch, active_masks_batch
        )

        # ============================== Total Loss ================================        
        total_loss = policy_loss + value_loss * self._policy.custom_config.get("value_loss_coef",1.0)
        
        total_loss = total_loss/self.grad_accum_step

        # ============================== Optimizer ================================
        total_loss.backward()        
        if self.n_opt_steps%self.grad_accum_step==0: 
            if self._use_max_grad_norm:
                for param_group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.max_grad_norm
                    )
            
            # Check for unusued parameters in the optimizer's param groups
            for pg in self.optimizer.param_groups:
                for p in pg['params']:
                    if p.grad is None:
                        Logger.info(f'{pg["name"]} - {p}')
                        Logger.info("found unused param")

            self.optimizer.step()
            self.optimizer.zero_grad()

        # ============================== Update Exploration Bonus ================================
        # If the exploration bonus is used, then we update the exploration bonus
        if self.rnd_bonus is not None:
            # Similar to the intrinsic reward network, we update the exploration bonus network if the update frequency is met
            # and we are in the final ppo epoch for the current batch
            # if self.n_opt_steps % self._policy.custom_config["ppo_epoch"] == 0 and \
            #    current_epoch % self._policy.custom_config.get("exploration_bonus_update_freq", 1) == 0:
                
            # Get state observations
            observations = sample[EpisodeKey.CUR_STATE]

            # Normalize the observations
            if self._num_agents == 10:
                start_offset = 330
            elif self._num_agents == 5:
                start_offset = 210

            # Extract the global state
            observations = observations[:, start_offset+19:]

            # Take one for each agent and remove duplicates
            observations = observations[0::self._num_agents]

            # Normalize the observations
            observations = self.obs_normalizer.normalize_obs(observations)

            # Update the predictor network
            Logger.info("Updating RND predictor network")
            self.rnd_bonus.update_predictor(observations)

        # ============================== Statistics ================================
        if update_actor:
            # TODO(jh): miss active masks?
            stats = dict(
                ratio=float(imp_weights.detach().mean().cpu().numpy()),
                ratio_std=float(imp_weights.detach().std().cpu().numpy()),
                policy_loss=float(policy_loss.detach().cpu().numpy()),
                value_loss=float(value_loss.detach().cpu().numpy()),
                entropy=float(dist_entropy.detach().mean().cpu().numpy()),
                approx_kl=approx_kl,
            )

            # Get advantage and delta for the non-shaped original reward
            adv_targ = sample[EpisodeKey.ADVANTAGE]
            adv_targ = self._select_data_from_agent_ids(adv_targ, agent_ids)
            delta = sample["delta"]
            stats.update(basic_stats("advantages", adv_targ))
            stats.update(basic_stats("delta", delta))

            # Get the intrinsic reward if the intrinsic reward network is used
            if self.intrinsic_reward_network or self.rnd_bonus is not None:
                if EpisodeKey.RETURN + "_with_intrinsic" in sample.keys():
                    adv_targ_intrinsic = sample[EpisodeKey.ADVANTAGE + "_with_intrinsic"]
                    delta_intrinsic = sample["delta_with_intrinsic"]
                    if self.intrinsic_reward_network:
                        intrinsic_reward = sample["intrinsic_reward"]
                    if self.rnd_bonus:
                        exploration_bonus = sample["intrinsic_exploration"]
                else:
                    adv_targ_intrinsic = torch.tensor(0)
                    delta_intrinsic = torch.tensor(0)
                    if self.intrinsic_reward_network:
                        intrinsic_reward = torch.tensor(0)
                    if self.rnd_bonus:
                        exploration_bonus = torch.tensor(0)
                
                stats.update(basic_stats("advantages_intrinsic", adv_targ_intrinsic))
                stats.update(basic_stats("delta_intrinsic", delta_intrinsic))
                if self.intrinsic_reward_network:
                    stats.update(basic_stats("intrinsic_reward", intrinsic_reward))
                if self.rnd_bonus:
                    stats.update(basic_stats("exploration_bonus", exploration_bonus))

            stats.update(basic_stats("imp_weights", imp_weights))
            stats.update(basic_stats("V", values))
            stats.update(basic_stats("Old_V", value_preds_batch))

            stats["upper_clip_ratio"] = to_value(
                (imp_weights > (1 + self.clip_param)).float().mean()
            )
            stats["lower_clip_ratio"] = to_value(
                (imp_weights < (1 - self.clip_param)).float().mean()
            )
            stats["clip_ratio"] = stats["upper_clip_ratio"] + stats["lower_clip_ratio"]

            stats["approx_kl"] = approx_kl

        else:
            stats = {}
            
        return stats

    def _calc_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch=None
    ):
        # Normalize the returns before feeding them into critic
        return_batch = self._policy.critic.module.model.running_stats.normalize(return_batch)

        # Make sure return_batch is on the same device as values
        return_batch = return_batch.to(values.device)

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if active_masks_batch is not None:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / (active_masks_batch.sum()+1e-20)
        else:
            value_loss = value_loss.mean()

        return value_loss

    def zero_grad(self):
        pass

    def step(self):
        pass
