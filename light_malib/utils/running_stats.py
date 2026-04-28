# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch

class RunningStats:
    def __init__(self, device='cpu'):
        self.n = 0
        self.mean = torch.tensor(0.0, device=device)
        self.M2 = torch.tensor(0.0, device=device)
        self.current_max = torch.tensor(float('-inf'), device=device)
        self.device = device

    def update(self, x):
        """Update the running statistics with a new tensor of values."""
        x = x.to(self.device)
        batch_size = x.shape[0]
       
        # Update the count
        self.n += batch_size
        
        # Update mean and M2
        delta = x - self.mean
        self.mean += delta.sum() / self.n
        delta2 = x - self.mean
        self.M2 += (delta * delta2).sum()

        # Update max
        self.current_max = torch.max(self.current_max, x.max())

    @property
    def max(self):
        """Return the maximum value of the accumulated values."""
        return self.current_max

    @property
    def variance(self):
        """Return the variance of the accumulated values."""
        return self.M2 / self.n if self.n > 1 else torch.tensor(0.0, device=self.device)

    @property
    def std(self):
        """Return the standard deviation of the accumulated values."""
        return torch.sqrt(self.variance)

    @property
    def avg(self):
        """Return the mean of the accumulated values."""
        return self.mean

    def normalize(self, x):
        """Normalize a tensor based on the running statistics."""
        x = x.to(self.device)
        if self.std > 0:
            return (x - self.mean) / self.std
        else:
            return x - self.mean

    def denormalize(self, x):
        """Denormalize a tensor based on the running statistics."""
        x = x.to(self.device)
        return x * self.std + self.mean if self.std > 0 else x + self.mean
