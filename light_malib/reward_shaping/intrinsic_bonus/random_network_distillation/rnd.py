# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch

from .utils import RunningMeanStd

# Random Network Distillation (RND) Intrinsic Bonus
class RNDIntrinsicBonus():
    def __init__(self, target, predictor, bonus_coef=1., training_batch_portion=1., lr=1e-4):
        self.target = target
        self.predictor = predictor
        self.bonus_coef = bonus_coef
        self.training_batch_portion = training_batch_portion
       
        self.loss_fn = torch.nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    # Compute the intrinsic reward from the normalized observations
    def compute_intrinsic_reward(self, obs):
        # Change obs dtype from float64 (double) to float32
        obs = obs.float()
        pred = self.predictor(obs)
        target = self.target(obs)
        intrinsic_reward = (pred - target).pow(2).sum(1) * self.bonus_coef
        return intrinsic_reward
    
    # Update the predictor network 
    def update_predictor(self, obs):
        # Change obs dtype from float64 (double) to float32
        obs = obs.float()
        target = self.target(obs).detach()
        self.optimizer.zero_grad()
        prediction = self.predictor.forward(obs)  
        loss = self.loss_fn(prediction, target)  
        loss.backward()
        self.optimizer.step()