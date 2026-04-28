# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from .base import MetaSolver
import numpy as np
from enum import Enum

class SelfPlayStage(Enum):  
    CURRICULUM = 1  
    CHALLENGE = 2
    GENERALIZE = 3  
    TEST_SINGLE_SCENARIO = 4

class Solver(MetaSolver):
    def __init__(self):
        self.iterations = 1

    def compute_curriculum(self, payoff_entry, agent_ids, current_diff):
        # Set the probability of the last policy as 1.0 and the rest as 0.0
        # i.e. choose the best version of policy in the last scenario as the opponent
        new_dist = [0.0 for _ in payoff_entry]

        # Check if we only have two opponent policy candidates
        # Then it means that we are in the first iteration, and we choose index 0, i.e. initial policy which is usually the build-in
        if len(new_dist) == 2 and "built_in" in agent_ids[0]:
            new_dist[0] = 1.0

        # Otherwise, use the last trained policy
        else:
            for i, agent_id in enumerate(agent_ids):
                if f'agent_0-default-{current_diff - 1}' in agent_id:
                    new_dist[i] = 1.0
                    break

        dists = (np.array(new_dist), np.array(new_dist))
        return dists
    
    def compute_challenge(self, payoff_entry):  
        # Set opponent as most recent agent with 80% probability.  
        # Else sample from the ones before (excluding the most recent agent) uniformly.  
        
        new_dist = [0.0 for _ in payoff_entry]  
    
        # Set the probability of the most recent agent to 0.8  
        new_dist[-1] = 0.8  

        # Calculate the remaining probability for other agents  
        remaining_probability = 0.2  
        num_remaining_agents = len(new_dist) - 1  # Exclude the most recent agent  
        remaining_probability_per_agent = remaining_probability / num_remaining_agents  # NOTE: May have to add a small number to avoid zero

        # Assign the remaining probability to all the agents except for the most recent agent  
        for i in range(num_remaining_agents):  
            new_dist[i] = remaining_probability_per_agent  

        dists = (np.array(new_dist), np.array(new_dist))  
        return dists 
    
    def compute_generalize(self, payoff_entry):  
        # Prioitise Fictitious Self-Play on win rate

        print("---------------------computing PFSP---------------------")
        newest_payoff_entry = payoff_entry
        fn = lambda x: (1 - x) ** 0.5
        fn_payoff_entry = [fn(x) for x in newest_payoff_entry]
        sum_fn = sum(fn_payoff_entry)
        PFSP_dist = [i / sum_fn for i in fn_payoff_entry]

        eqs = (np.array(PFSP_dist), np.array(PFSP_dist))
        return eqs
    
    def choose_initial_policy(self, agent_ids):
        new_dist = [0.0 for _ in agent_ids]
        new_dist[0] = 1.0

        dists = (np.array(new_dist), np.array(new_dist))
        return dists 


    def compute(self, payoff, agent_ids, mode=SelfPlayStage.CURRICULUM, current_diff=0):
        """
        Meta-solver based on TiZero's opponent selection scheme
        """
        newest_payoff_entry = payoff[-1, :]

        dists = ()
        if mode is SelfPlayStage.CURRICULUM:
            dists = self.compute_curriculum(newest_payoff_entry, agent_ids, current_diff)
        elif mode is SelfPlayStage.CHALLENGE:
            dists = self.compute_challenge(newest_payoff_entry)
        elif mode is SelfPlayStage.GENERALIZE:
            dists = self.compute_generalize(newest_payoff_entry)
        elif mode is SelfPlayStage.TEST_SINGLE_SCENARIO:
            dists = self.choose_initial_policy(agent_ids)
        
        return dists
