# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from typing import OrderedDict
from light_malib.registry import registry
from light_malib.agent.agent_manager import AgentManager

from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.utils.logger import Logger
from light_malib.agent import Population
from light_malib.utils.desc.task_desc import TrainingDesc
from light_malib.framework.meta_solver.tizero import Solver, SelfPlayStage
import numpy as np
import importlib


class CurriculumScheduler:
    def __init__(
        self, cfg, agent_manager: AgentManager, policy_data_manager: PolicyDataManager
    ):
        self.cfg = cfg
        self.agent_manager = agent_manager
        self.agents = self.agent_manager.agents
        self.population_id = "default"
        self.policy_data_manager = policy_data_manager
        # self.meta_solver_type = self.cfg.get("meta_solver", "tizero")
        self.sync_training = self.cfg.get("sync_training", False)

        Logger.warning("use meta solver type: tizero")
        #solver_module = importlib.import_module(
        #    "light_malib.framework.meta_solver.{}".format(self.meta_solver_type)
        #)
        self.meta_solver = Solver()
        self._schedule = self._gen_schedule()

    def initialize(self, populations_cfg):
        # Add Population
        Logger.info(f'Initalizing population in CurriculumScheduler: going through training agent ids {self.agents.training_agent_ids}')
        for agent_id in self.agents.training_agent_ids:
            assert len(populations_cfg) == 1
            population_id = populations_cfg[0]["population_id"]
            assert population_id == self.population_id
            algorithm_cfg = populations_cfg[0]["algorithm"]
            self.agent_manager.add_new_population(
                agent_id, self.population_id, algorithm_cfg
            )

        for population_cfg in populations_cfg:
            population_id = population_cfg["population_id"]
            algorithm_cfg = population_cfg.algorithm
            policy_init_cfg = algorithm_cfg.get("policy_init_cfg", None)
            if policy_init_cfg is None:
                continue
            for agent_id, agent_policy_init_cfg in policy_init_cfg.items():
                agent_initial_policies = agent_policy_init_cfg.get(
                    "initial_policies", None
                )
                if agent_initial_policies is None:
                    continue
                for policy_cfg in agent_initial_policies:
                    policy_id = policy_cfg["policy_id"]
                    policy_dir = policy_cfg["policy_dir"]
                    self.agent_manager.load_policy(
                        agent_id, population_id, policy_id, policy_dir
                    )
                    Logger.info(f"Load initial policy {policy_id} from {policy_dir}")

        # generate the first policy
        for agent_id in self.agents.training_agent_ids:
            self.agent_manager.gen_new_policy(agent_id, self.population_id)

        Logger.warning("After CurriculumScheduler initialization:\n{}".format(self.agents))

    def _gen_schedule(self):
        
        generation_ctr = self.cfg.curriculum.start_diff
        while True:
            # Get the training agent id
            assert len(self.agents.training_agent_ids) == 1, "Training more than one agent is not supported yet"
            training_agent_id = self.agents.training_agent_ids[0]

            generation_ctr += 1

            # Get all available policy_ids from the population for all agents in the game
            agent_id2policy_ids = OrderedDict()
            agent_id2policy_indices = OrderedDict()
            for agent_id in self.agents.keys():
                population: Population = self.agents[agent_id].populations[
                    self.population_id
                ]
                agent_id2policy_ids[agent_id] = population.policy_ids
                agent_id2policy_indices[agent_id] = np.array(
                    [
                        self.agents[agent_id].policy_id2idx[policy_id]
                        for policy_id in population.policy_ids
                    ]
                )

            # Get payoff matrix
            payoff_matrix = self.policy_data_manager.get_matrix_data(
                "payoff", agent_id2policy_indices
            )

            # Compute distributions for the policies of each agent
            computed_policy_dists = self.meta_solver.compute(payoff_matrix, agent_id2policy_ids['agent_0'], mode=SelfPlayStage.CURRICULUM, current_diff=generation_ctr)

            policy_distributions = {}
            for probs, (agent_id, policy_ids) in zip(
                computed_policy_dists, agent_id2policy_ids.items()
            ):
                policy_distributions[agent_id] = OrderedDict(zip(policy_ids, probs))

            # Generate a new policy for the training agent
            training_policy_id = self.agent_manager.gen_new_policy(
                training_agent_id, self.population_id
            )
            policy_distributions[training_agent_id] = {training_policy_id: 1.0}

            Logger.info(f'Current Policy Distributions: {policy_distributions}')

            Logger.warning(
                "********** Generation[{}] Agent[{}] START **********".format(
                    generation_ctr, training_agent_id
                )
            )

            # Increase the win-rate from 55% to 75% so that at 0 diff, the win-rate is 55% and at diff 8 it is 75%
            increment = (self.cfg.stopper.kwargs.curriculum_min_win_rate - 0.55) / (8 - 0)
            win_rate_threshold = 0.55 + increment * (generation_ctr - 1)
            win_rate_threshold = min(win_rate_threshold, self.cfg.stopper.kwargs.curriculum_min_win_rate)

            stopper = registry.get(registry.STOPPER, "curriculum_win_rate_stopper")(
                policy_data_manager=self.policy_data_manager,
                **{'curriculum_min_win_rate': win_rate_threshold},
            )

            training_desc = TrainingDesc(
                training_agent_id,
                training_policy_id,
                policy_distributions,
                self.agents.share_policies,
                self.sync_training,
                stopper,
                False,
                False,
            )
            
            yield training_desc

    def get_task(self):
        try:
            task = next(self._schedule)
            return task
        except StopIteration:
            return None

    def submit_result(self, result):
        pass

class ChallengeScheduler:
    def __init__(
        self, cfg, agent_manager: AgentManager, policy_data_manager: PolicyDataManager
    ):
        self.cfg = cfg
        self.agent_manager = agent_manager
        self.agents = self.agent_manager.agents
        self.population_id = "default"
        self.policy_data_manager = policy_data_manager
        self.sync_training = self.cfg.get("sync_training", False)

        Logger.warning("use meta solver type: tizero")
        self.meta_solver = Solver()
        self._schedule = self._gen_schedule()


    # NOTE: No initialization needed since the agent_manager is initialized with the populations_cfg in CurriculumScheduler
    def initialize(self, populations_cfg):
        # Add Population
        Logger.info(f'Initalizing ChallengeScheduler')

    def _gen_schedule(self):
        
        generation_ctr = 0
        while True:
            # Get the training agent id
            assert len(self.agents.training_agent_ids) == 1, "Training more than one agent is not supported yet"
            training_agent_id = self.agents.training_agent_ids[0]

            generation_ctr += 1

            # Get all available policy_ids from the population for all agents in the game
            agent_id2policy_ids = OrderedDict()
            agent_id2policy_indices = OrderedDict()
            for agent_id in self.agents.keys():
                population: Population = self.agents[agent_id].populations[
                    self.population_id
                ]
                agent_id2policy_ids[agent_id] = population.policy_ids
                agent_id2policy_indices[agent_id] = np.array(
                    [
                        self.agents[agent_id].policy_id2idx[policy_id]
                        for policy_id in population.policy_ids
                    ]
                )

            # Get payoff matrix
            payoff_matrix = self.policy_data_manager.get_matrix_data(
                "payoff", agent_id2policy_indices
            )

            # Compute distributions for the policies of each agent
            computed_policy_dists = self.meta_solver.compute(payoff_matrix, agent_id2policy_ids['agent_0'], mode=SelfPlayStage.CHALLENGE)

            policy_distributions = {}
            for probs, (agent_id, policy_ids) in zip(
                computed_policy_dists, agent_id2policy_ids.items()
            ):
                policy_distributions[agent_id] = OrderedDict(zip(policy_ids, probs))

            # Generate a new policy for the training agent
            training_policy_id = self.agent_manager.gen_new_policy(
                training_agent_id, self.population_id
            )
            policy_distributions[training_agent_id] = {training_policy_id: 1.0}

            Logger.info(f'Current Policy Distributions: {policy_distributions}')

            Logger.warning(
                "********** Generation[{}] Agent[{}] START **********".format(
                    generation_ctr, training_agent_id
                )
            )

            stopper = registry.get(registry.STOPPER, "challenge_win_rate_stopper")(
                policy_data_manager=self.policy_data_manager,
                **self.cfg.stopper.kwargs,
            )

            training_desc = TrainingDesc(
                training_agent_id,
                training_policy_id,
                policy_distributions,
                self.agents.share_policies,
                self.sync_training,
                stopper,
                False,
                False,
            )
            
            yield training_desc

    def get_task(self):
        try:
            task = next(self._schedule)
            return task
        except StopIteration:
            return None

    def submit_result(self, result):
        pass

class GeneralizeScheduler:
    def __init__(
        self, cfg, agent_manager: AgentManager, policy_data_manager: PolicyDataManager
    ):
        self.cfg = cfg
        self.agent_manager = agent_manager
        self.agents = self.agent_manager.agents
        self.population_id = "default"
        self.policy_data_manager = policy_data_manager
        self.sync_training = self.cfg.get("sync_training", False)

        Logger.warning("use meta solver type: tizero")
        self.meta_solver = Solver()
        self._schedule = self._gen_schedule()


    # NOTE: No initialization needed since the agent_manager is initialized with the populations_cfg in CurriculumScheduler
    def initialize(self, populations_cfg):
        # Add Population
        Logger.info(f'Initalizing GeneralizeScheduler')

    def _gen_schedule(self):
        
        generation_ctr = 0
        while True:
            # Get the training agent id
            assert len(self.agents.training_agent_ids) == 1, "Training more than one agent is not supported yet"
            training_agent_id = self.agents.training_agent_ids[0]

            generation_ctr += 1

            # Get all available policy_ids from the population for all agents in the game
            agent_id2policy_ids = OrderedDict()
            agent_id2policy_indices = OrderedDict()
            for agent_id in self.agents.keys():
                population: Population = self.agents[agent_id].populations[
                    self.population_id
                ]
                agent_id2policy_ids[agent_id] = population.policy_ids
                agent_id2policy_indices[agent_id] = np.array(
                    [
                        self.agents[agent_id].policy_id2idx[policy_id]
                        for policy_id in population.policy_ids
                    ]
                )

            # Get payoff matrix
            payoff_matrix = self.policy_data_manager.get_matrix_data(
                "payoff", agent_id2policy_indices
            )

            # Compute distributions for the policies of each agent
            computed_policy_dists = self.meta_solver.compute(payoff_matrix, agent_id2policy_ids['agent_0'], mode=SelfPlayStage.GENERALIZE)

            policy_distributions = {}
            for probs, (agent_id, policy_ids) in zip(
                computed_policy_dists, agent_id2policy_ids.items()
            ):
                policy_distributions[agent_id] = OrderedDict(zip(policy_ids, probs))

            # Generate a new policy for the training agent
            training_policy_id = self.agent_manager.gen_new_policy(
                training_agent_id, self.population_id
            )
            policy_distributions[training_agent_id] = {training_policy_id: 1.0}

            Logger.info(f'Current Policy Distributions: {policy_distributions}')

            Logger.warning(
                "********** Generation[{}] Agent[{}] START **********".format(
                    generation_ctr, training_agent_id
                )
            )

            stopper = registry.get(registry.STOPPER, "generalize_win_rate_stopper")(
                policy_data_manager=self.policy_data_manager,
                **self.cfg.stopper.kwargs,
            )

            training_desc = TrainingDesc(
                training_agent_id,
                training_policy_id,
                policy_distributions,
                self.agents.share_policies,
                self.sync_training,
                stopper,
                False,
                False,
            )
            
            yield training_desc

    def get_task(self):
        try:
            task = next(self._schedule)
            return task
        except StopIteration:
            return None

    def submit_result(self, result):
        pass

class SingleScenarioScheduler:
    def __init__(
        self, cfg, agent_manager: AgentManager, policy_data_manager: PolicyDataManager
    ):
        self.cfg = cfg
        self.agent_manager = agent_manager
        self.agents = self.agent_manager.agents
        self.population_id = "default"
        self.policy_data_manager = policy_data_manager
        self.sync_training = self.cfg.get("sync_training", False)

        self.meta_solver = Solver()
        self._schedule = self._gen_schedule()

    def initialize(self, populations_cfg):
        # Add Population
        Logger.info(f'Initalizing population in SingleScenarioScheduler: going through training agent ids {self.agents.training_agent_ids}')
        for agent_id in self.agents.training_agent_ids:
            assert len(populations_cfg) == 1
            population_id = populations_cfg[0]["population_id"]
            assert population_id == self.population_id
            algorithm_cfg = populations_cfg[0]["algorithm"]
            self.agent_manager.add_new_population(
                agent_id, self.population_id, algorithm_cfg
            )

        for population_cfg in populations_cfg:
            population_id = population_cfg["population_id"]
            algorithm_cfg = population_cfg.algorithm
            policy_init_cfg = algorithm_cfg.get("policy_init_cfg", None)
            if policy_init_cfg is None:
                continue
            for agent_id, agent_policy_init_cfg in policy_init_cfg.items():
                agent_initial_policies = agent_policy_init_cfg.get(
                    "initial_policies", None
                )
                if agent_initial_policies is None:
                    continue
                for policy_cfg in agent_initial_policies:
                    policy_id = policy_cfg["policy_id"]
                    policy_dir = policy_cfg["policy_dir"]
                    self.agent_manager.load_policy(
                        agent_id, population_id, policy_id, policy_dir
                    )
                    Logger.info(f"Load initial policy {policy_id} from {policy_dir}")

        # generate the first policy
        for agent_id in self.agents.training_agent_ids:
            self.agent_manager.gen_new_policy(agent_id, self.population_id)

        Logger.warning("After SingleScenarioScheduler initialization:\n{}".format(self.agents))

    def _gen_schedule(self):
        
        generation_ctr = 0
        while True:
            # Get the training agent id
            assert len(self.agents.training_agent_ids) == 1, "Training more than one agent is not supported yet"
            training_agent_id = self.agents.training_agent_ids[0]

            generation_ctr += 1

            # Get all available policy_ids from the population for all agents in the game
            agent_id2policy_ids = OrderedDict()
            agent_id2policy_indices = OrderedDict()
            for agent_id in self.agents.keys():
                population: Population = self.agents[agent_id].populations[
                    self.population_id
                ]
                agent_id2policy_ids[agent_id] = population.policy_ids
                agent_id2policy_indices[agent_id] = np.array(
                    [
                        self.agents[agent_id].policy_id2idx[policy_id]
                        for policy_id in population.policy_ids
                    ]
                )

            # Get payoff matrix
            payoff_matrix = self.policy_data_manager.get_matrix_data(
                "payoff", agent_id2policy_indices
            )

            # Compute distributions for the policies of each agent
            computed_policy_dists = self.meta_solver.compute(payoff_matrix, agent_id2policy_ids["agent_0"], mode=SelfPlayStage.TEST_SINGLE_SCENARIO)

            policy_distributions = {}
            for probs, (agent_id, policy_ids) in zip(
                computed_policy_dists, agent_id2policy_ids.items()
            ):
                policy_distributions[agent_id] = OrderedDict(zip(policy_ids, probs))

            # Generate a new policy for the training agent
            training_policy_id = self.agent_manager.gen_new_policy(
                training_agent_id, self.population_id
            )
            policy_distributions[training_agent_id] = {training_policy_id: 1.0}

            Logger.info(f'Current Policy Distributions: {policy_distributions}')

            Logger.warning(
                "********** Generation[{}] Agent[{}] START **********".format(
                    generation_ctr, training_agent_id
                )
            )

            stopper = registry.get(registry.STOPPER, "challenge_win_rate_stopper")(
                policy_data_manager=self.policy_data_manager,
                **self.cfg.stopper.kwargs,
            )

            training_desc = TrainingDesc(
                training_agent_id,
                training_policy_id,
                policy_distributions,
                self.agents.share_policies,
                self.sync_training,
                stopper,
                False,
                False,
            )
            
            yield training_desc

    def get_task(self):
        try:
            task = next(self._schedule)
            return task
        except StopIteration:
            return None

    def submit_result(self, result):
        pass
