# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import os

from collections import OrderedDict
from tokenize import Single
from omegaconf import OmegaConf

from light_malib import rollout, training, buffer
from light_malib.agent import AgentManager
from light_malib.evaluation.evaluation_manager import EvaluationManager
from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.framework.scheduler.tizero_schedulers import CurriculumScheduler, ChallengeScheduler, GeneralizeScheduler, SingleScenarioScheduler

import ray
import numpy as np
from easydict import EasyDict
from light_malib.utils.distributed import get_resources
from light_malib.utils.logger import Logger, LOG_LEVEL

# Set the logger.disabled to false to enable the logger and set level to logging.INFO
Logger.disabled = False
Logger.setLevel(LOG_LEVEL)

class TiZeroRunner:
    def __init__(self, cfg, cfg_raw):
        self.cfg_raw = cfg_raw
        self.cfg = cfg
        self.framework_cfg = self.cfg.framework
        self.id = self.framework_cfg.name
        self.test_mode = self.framework_cfg.test_mode

        self.rew_cfg = self.cfg.rollout_manager.worker.envs[0].reward_config

        # NOTE: (important) The default scenario configuration for the session is partially hardcoded here,
        # and if these are changed in the conifguration file, they should be updated here as well.
        self.default_scenario_cfg = EasyDict(
            {
                "envs": {
                    "cls": "gr_football",  
                    "id_prefix": "gr_footbal",  
                    "scenario_config": {  
                        "env_name": self.cfg.rollout_manager.worker.envs[0].scenario_config["env_name"],  
                        "number_of_left_players_agent_controls": 10,  
                        "number_of_right_players_agent_controls": 10,  
                        "representation": "raw",
                        "rewards": "scoring",  
                        "stacked": False,
                        "logdir": "/tmp/football/malib_psro",  
                        "write_goal_dumps": False,  
                        "write_full_episode_dumps": False,  
                        "render": False,  
                        "other_config_options": {  
                            "action_set": "v2"
                        },  
                    },  
                    "reward_config": {
                        # "repeated_action_penalty": self.rew_cfg.repeated_action_penalty,
                        "out_of_bound_penalty": self.rew_cfg.out_of_bound_penalty,
                        "gather_penalty": self.rew_cfg.gather_penalty,
                        "pass_reward": self.rew_cfg.pass_reward,
                        "hold_ball": self.rew_cfg.hold_ball,
                        "win_reward": self.rew_cfg.win_reward if hasattr(self.rew_cfg, "win_reward") else None,
                        "yellow_reward": self.rew_cfg.yellow_reward if hasattr(self.rew_cfg, "yellow_reward") else None,
                        "min_dist_reward": self.rew_cfg.min_dist_reward if hasattr(self.rew_cfg, "min_dist_reward") else None,
                        "total_dist_to_ball_reward": self.rew_cfg.total_dist_to_ball_reward if hasattr(self.rew_cfg, "total_dist_to_ball_reward") else None,
                        "different_action_reward": self.rew_cfg.different_action_reward if hasattr(self.rew_cfg, "different_action_reward") else None,
                        "goal_reward": self.rew_cfg.goal_reward if hasattr(self.rew_cfg, "goal_reward") else None,
                        "official_reward": self.rew_cfg.official_reward,  
                    },
                }
            }
        )

        ###### Initialize Components #####
        RolloutManager = ray.remote(
            **get_resources(cfg.rollout_manager.distributed.resources)
        )(rollout.RolloutManager)
        TrainingManager = ray.remote(
            **get_resources(cfg.training_manager.distributed.resources)
        )(training.TrainingManager)
        DataServer = ray.remote(**get_resources(cfg.data_server.distributed.resources))(
            buffer.DataServer
        )
        PolicyServer = ray.remote(
            **get_resources(cfg.policy_server.distributed.resources)
        )(buffer.PolicyServer)

        # Create Agents
        agents = AgentManager.build_agents(self.cfg.agent_manager)

        self.data_server = DataServer.options(
            name="DataServer", max_concurrency=self.cfg.rollout_manager.num_workers+5
        ).remote("DataServer", self.cfg.data_server)

        self.policy_server = PolicyServer.options(
            name="PolicyServer", max_concurrency=self.cfg.rollout_manager.num_workers+5
        ).remote("PolicyServer", self.cfg.policy_server, agents)

        self.rollout_manager = RolloutManager.options(
            name="RolloutManager", max_concurrency=self.cfg.rollout_manager.num_workers+5
        ).remote("RolloutManager", self.cfg.rollout_manager, agents)

        self.training_manager = TrainingManager.options(
            name="TrainingManager", max_concurrency=5
        ).remote("TrainingManager", self.cfg.training_manager)

        Logger.info("Setup all the components")
        
        self.agent_manager = AgentManager(self.cfg.agent_manager)
        self.policy_data_manager = PolicyDataManager(
            self.cfg.policy_data_manager, self.agent_manager
        )
        self.evaluation_manager = EvaluationManager(
            self.cfg.evaluation_manager, self.agent_manager, self.policy_data_manager
        )

        self.reached_max_rollout_steps = False

        if self.id == "tizero":
            # Regular TiZero with three stages
            if not self.test_mode:
                self.curriculum_scheduler = CurriculumScheduler(
                    self.cfg.framework, self.agent_manager, self.policy_data_manager
                )
                self.challenge_scheduler = ChallengeScheduler(
                    self.cfg.framework, self.agent_manager, self.policy_data_manager
                )
                self.generalize_scheduler = GeneralizeScheduler(
                    self.cfg.framework, self.agent_manager, self.policy_data_manager
                )
            # Test mode where only one scenario is used and the agents train indefinitely, for testing and validation purposes
            else: 
                self.test_scheduler = SingleScenarioScheduler(
                    self.cfg.framework, self.agent_manager, self.policy_data_manager
                )
        else:
            raise NotImplementedError

        Logger.info("TiZeroRunner {} initialized".format(self.id))

    # Save config file
    def save_cfg(self, path):
        with open(path, "w") as f:
            cfg_omega = OmegaConf.create(self.cfg_raw)  
            f.write(OmegaConf.to_yaml(cfg_omega))  


    # Curriculum Self-Play 
    def curriculum_self_play(self):
        Logger.info(f'Curriculum self-play for TiZeroRunner {self.id} initiated')

        # Initialize the first policy and a builtin one for the start of the scenarios
        self.curriculum_scheduler.initialize(self.cfg.populations)

        # Reset the rollout length of the rollout manager and workers
        ray.get(self.rollout_manager.reset_rollout_length.remote(self.framework_cfg.curriculum.rollout_length))

        self.max_diff_lvl = self.framework_cfg.curriculum.max_diff_lvl
        current_diff_lvl = self.framework_cfg.curriculum.start_diff

        while current_diff_lvl < self.max_diff_lvl:
            # Initialize the scenario with the right level of difficulty
            current_scenario_cfg = self.default_scenario_cfg
            current_scenario_cfg["envs"]["scenario_config"]["env_name"] = f'full_game_10_vs_10_diff_{current_diff_lvl}'
            current_scenario_cfg["envs"]["scenario_config"]["render"] = self.cfg.rollout_manager.worker.envs[0].scenario_config.render

            ray.get(self.rollout_manager.initialize_workers_env.remote(current_scenario_cfg))

            Logger.info(f'Initialized the scenario for difficulty {current_diff_lvl} in the rollout manager')

            training_desc = self.curriculum_scheduler.get_task()
            if training_desc is None:
                break

            # If we are to use the intrinsic reward (SSIR) in the curriculum phase and the exploration bonus (RND) in the remaining phases,
            # update the training_desc accordingly
            if self.cfg.training_manager.use_intrinsic_first_then_exploration_bonus:
                training_desc.use_intrinsic_reward = True
                training_desc.use_exploration_bonus = False
            else:
                training_desc.use_intrinsic_reward = self.cfg.training_manager.trainer.use_intrinsic_reward
                training_desc.use_exploration_bonus = self.cfg.training_manager.trainer.use_exploration_bonus

            Logger.info("training_desc: {}".format(training_desc))
            training_task_ref = self.training_manager.train.remote(training_desc)
            global_rollout_steps = ray.get(training_task_ref) 

            if self.cfg.rollout_manager.max_global_rollout_steps != -1 and global_rollout_steps >= self.cfg.rollout_manager.max_global_rollout_steps:
                self.reached_max_rollout_steps = True
                break

            self.curriculum_scheduler.submit_result(None)

            current_diff_lvl += 1

            # Save the new config file with a new start diff and a new "==0" rule to support resuming from another difficulty
            self.cfg_raw["framework"]["curriculum"]["start_diff"] = current_diff_lvl
            self.cfg_raw["populations"][0]["algorithm"]["policy_init_cfg"]["agent_0"]["init_cfg"][0]["strategy"] = "pretrained"
            self.cfg_raw["populations"][0]["algorithm"]["policy_init_cfg"]["agent_0"]["init_cfg"][0]["policy_id"] = "best_curr"
            self.cfg_raw["populations"][0]["algorithm"]["policy_init_cfg"]["agent_0"]["init_cfg"][0]["policy_dir"] = os.path.join(self.cfg.expr_log_dir, f'agent_0/agent_0-default-{current_diff_lvl}')
            yaml_path = os.path.join(self.cfg.expr_log_dir, "config.yaml")
            self.save_cfg(yaml_path)

        Logger.info(f'Curriculum self-play for TiZeroRunner {self.id} ended')

    # Challenge Self-Play
    def challenge_self_play(self):
        Logger.info(f'Challenge self-play for TiZeroRunner {self.id} initiated')

        self.challenge_scheduler.initialize(self.cfg.populations)

        # Reset the rollout length of the rollout manager and workers
        ray.get(self.rollout_manager.reset_rollout_length.remote(self.framework_cfg.challenge.rollout_length))

        # Initialize the regular 5v5 scenario (difficulty 10 of the 5v5 full game scenario)
        challenge_scenario = self.default_scenario_cfg
        challenge_scenario["envs"]["scenario_config"]["env_name"] = f'full_game_10_vs_10_challenge'
        challenge_scenario["envs"]["scenario_config"]["render"] = self.cfg.rollout_manager.worker.envs[0].scenario_config.render

        # Initialize the rollout workers
        ray.get(self.rollout_manager.initialize_workers_env.remote(challenge_scenario))

        Logger.info("Initialized the challenge scenario for Challenge Self-Play")

        # One round of Challenge Self-Play, done when win-rate threshold is reached

        # TODO: This evaluation could also be possibly removed to save some training time, as the policy probs are calculated
        # regardless of the evaluation results...
        self.evaluation_manager.eval()

        training_desc = self.challenge_scheduler.get_task()

        # If we are to use the intrinsic reward (SSIR) in the curriculum phase and the exploration bonus (RND) in the remaining phases,
        # update the training_desc accordingly
        if self.cfg.training_manager.use_intrinsic_first_then_exploration_bonus:
            training_desc.use_intrinsic_reward = False
            training_desc.use_exploration_bonus = True
        else:
            training_desc.use_intrinsic_reward = self.cfg.training_manager.trainer.use_intrinsic_reward
            training_desc.use_exploration_bonus = self.cfg.training_manager.trainer.use_exploration_bonus

        Logger.info("training_desc: {}".format(training_desc))

        training_task_ref = self.training_manager.train.remote(training_desc)

        global_rollout_steps = ray.get(training_task_ref)

        if self.cfg.rollout_manager.max_global_rollout_steps != -1 and global_rollout_steps >= self.cfg.rollout_manager.max_global_rollout_steps:
            self.reached_max_rollout_steps = True

        self.challenge_scheduler.submit_result(None)

        Logger.info(f'Challenge self-play for TiZeroRunner {self.id} ended')
    
    # Generalize Self-Play
    def generalize_self_play(self):
        Logger.info(f'Generalize self-play for TiZeroRunner {self.id} initiated')

        self.generalize_scheduler.initialize(self.cfg.populations)

        # Reset the rollout length of the rollout manager and workers
        ray.get(self.rollout_manager.reset_rollout_length.remote(self.framework_cfg.generalize.rollout_length))

        # Initialize the regular 5v5 scenario (difficulty 10 of the 5v5 full game scenario)
        generalize_scenario = self.default_scenario_cfg
        generalize_scenario["envs"]["scenario_config"]["env_name"] = f'full_game_10_vs_10_challenge'
        generalize_scenario["envs"]["scenario_config"]["render"] = self.cfg.rollout_manager.worker.envs[0].scenario_config.render

        # Initialize the rollout workers
        ray.get(self.rollout_manager.initialize_workers_env.remote(generalize_scenario))

        Logger.info("Initialized the generalize scenario for Challenge Self-Play")

        # One round of Generalize Self-Play, done when win-rate threshold is reached

        self.evaluation_manager.eval()

        training_desc = self.generalize_scheduler.get_task()

        # If we are to use the intrinsic reward (SSIR) in the curriculum phase and the exploration bonus (RND) in the remaining phases,
        # update the training_desc accordingly
        if self.cfg.training_manager.use_intrinsic_first_then_exploration_bonus:
            training_desc.use_intrinsic_reward = False
            training_desc.use_exploration_bonus = True
        else:
            training_desc.use_intrinsic_reward = self.cfg.training_manager.trainer.use_intrinsic_reward
            training_desc.use_exploration_bonus = self.cfg.training_manager.trainer.use_exploration_bonus

        Logger.info("training_desc: {}".format(training_desc))

        training_task_ref = self.training_manager.train.remote(training_desc)

        global_rollout_steps = ray.get(training_task_ref)

        if self.cfg.rollout_manager.max_global_rollout_steps != -1 and global_rollout_steps >= self.cfg.rollout_manager.max_global_rollout_steps:
            self.reached_max_rollout_steps = True

        self.generalize_scheduler.submit_result(None)

        Logger.info(f'Generalize self-play for TiZeroRunner {self.id} ended')
        
    # Check if the current policy has converged or not
    def has_converged(self):
        pass

    def test_single_scenario(self):
        Logger.info(f'Test mode for TiZeroRunner {self.id} initiated')

        self.test_scheduler.initialize(self.cfg.populations)

        # Reset the rollout length of the rollout manager and workers
        ray.get(self.rollout_manager.reset_rollout_length.remote(self.framework_cfg.challenge.rollout_length))

        # Initialize the test 10v10 scenario
        challenge_scenario = self.default_scenario_cfg
        challenge_scenario["envs"]["scenario_config"]["env_name"] = f'test_scenario_10v10_go_to_ball'
        challenge_scenario["envs"]["scenario_config"]["render"] = self.cfg.rollout_manager.worker.envs[0].scenario_config.render

        # Initialize the rollout workers
        ray.get(self.rollout_manager.initialize_workers_env.remote(challenge_scenario))

        Logger.info("Initialized the test scenario for Single Scenario Test")

        training_desc = self.test_scheduler.get_task()

        Logger.info("training_desc: {}".format(training_desc))

        training_task_ref = self.training_manager.train.remote(training_desc)

        ray.get(training_task_ref)

        self.test_scheduler.submit_result(None)

        Logger.info(f'Single Scenario Test for TiZeroRunner {self.id} ended')       

    # Training Procedure
    def run(self):
        Logger.info(f'TiZeroRunner {self.id} started')
        # If evaluation
        if self.cfg.eval_only:
            self.evaluation_manager.eval(eval_more_metrics=True)

        # If testing on a single scenario
        elif self.test_mode:
            self.test_single_scenario()

        # If training
        else:
            # Curriculum Self-Play (Stage 1)
            self.curriculum_self_play()

            # Challenge and Generalize Self-Play (Stage 2)
            while True and not self.reached_max_rollout_steps:
                # Challenge SP
                self.challenge_self_play()

                if self.reached_max_rollout_steps:
                    break

                # Generalize SP
                self.generalize_self_play()

                # Check if we have converged
                # if self.has_converged():
                #     break

        Logger.info("TiZeroRunner {} ended".format(self.id))

    def close(self):
        ray.get(self.training_manager.close.remote())
        ray.get(self.rollout_manager.close.remote())
