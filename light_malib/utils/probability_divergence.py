# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch
import math

def logprobs_to_probs(logprobs):
    """
    Arguments:
        logprobs: Tensor, (batch_size, n_actions)
    
    Output:
        probs: Tensor, (batch_size, n_actions)
    """
    probs = torch.exp(logprobs)
    return probs

def KL_divergence(p, q):
    """
    Kullback-Leibler Divergence

    Arguments:
        p: Tensor, (batch_size, n_actions)
        q: Tensor, (batch_size, n_actions)
    
    Output:
        kl_div: Tensor, (batch_size, 1)
    """
    kl_div = torch.sum(p * torch.log(p / q), dim=1)
    return kl_div

def JS_divergence(prob_dists: torch.Tensor):
    """
    Jensen-Shannon Divergence

    Arguments:
        prob_dists: Tensor, (batch_size, n_actions)    
    
    Output:
        js_divergence: Tensor, (batch_size, 1)
    """
    # Number of distributions  
    n = prob_dists.shape[0]  
      
    # Calculate the mean distribution  
    mean_dist = torch.mean(prob_dists, dim=0).unsqueeze(0)  
      
    # Calculate the KL divergence for each distribution with the mean distribution  
    kl_divs = [KL_divergence(dist.unsqueeze(0), mean_dist) for dist in prob_dists]  
      
    # Average the KL divergences  
    jsd = torch.mean(torch.stack(kl_divs), dim=0) / 2  
      
    return jsd 
