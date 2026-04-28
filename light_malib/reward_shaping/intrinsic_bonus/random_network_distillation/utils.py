# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch  
import torch.distributed as dist
import copy
  
class RunningMeanStd(object):  
    def __init__(self, epsilon=1e-4, shape=()):  
        self.mean = torch.zeros(shape, dtype=torch.float64, device="cuda:0")  
        self.var = torch.ones(shape, dtype=torch.float64, device="cuda:0")  
        self.count = epsilon  
  
    def update(self, x):  
        batch_mean, batch_std, batch_count = x.mean(dim=0), x.std(dim=0), x.shape[0]  # change here  
        batch_var = batch_std.pow(2)  
        self.update_from_moments(batch_mean, batch_var, batch_count)  
  
    def update_from_moments(self, batch_mean, batch_var, batch_count):  
        delta = batch_mean - self.mean  
        tot_count = self.count + batch_count  
  
        new_mean = self.mean + delta * batch_count / tot_count  
        m_a = self.var * self.count  
        m_b = batch_var * batch_count  
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / (self.count + batch_count)  
        new_var = M2 / (self.count + batch_count)  
  
        new_count = batch_count + self.count  
  
        self.mean = new_mean  
        self.var = new_var  
        self.count = new_count  
  
    def normalize_obs(self, obs):
        normalized_obs = (obs - self.mean) / torch.sqrt(self.var + 1e-8)  
        clipped_obs = torch.clamp(normalized_obs, -5.0, 5.0)  
        return clipped_obs  
  
    def normalize_reward(self, reward):  
        normalized_reward = reward / torch.sqrt(self.var + 1e-8)  
        return normalized_reward
    
class DistributedRunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), device="cuda"):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float64, device=x.device)
        
        # Gather statistics from all processes
        dist.all_reduce(batch_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_var, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)

        # Calculate global statistics
        world_size = dist.get_world_size()
        batch_mean /= world_size
        batch_var /= world_size
        batch_count /= world_size
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize_obs(self, obs):
        normalized_obs = (obs - self.mean) / torch.sqrt(self.var + 1e-8)
        clipped_obs = torch.clamp(normalized_obs, -5.0, 5.0)
        return clipped_obs

    def normalize_reward(self, reward):
        normalized_reward = reward / torch.sqrt(self.var + 1e-8)
        return normalized_reward
    
    def to_device(self, device):
        clone = copy.deepcopy(self)
        clone.mean = self.mean.to(device)
        clone.var = self.var.to(device)
        clone.count = self.count.to(device)
        return clone
    