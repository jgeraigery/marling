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

import threading

from light_malib.utils.naming import default_table_name
from ..utils.desc.task_desc import PrefetchingDesc, RolloutDesc, TrainingDesc
from ..utils.distributed import get_actor, get_resources
from . import distributed_trainer
import ray
from ..utils.decorator import limited_calls
from . import data_prefetcher
import numpy as np
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_trainer_id
import os
import pickle
import torch

import socket

class TrainingManager:
    def __init__(self, id, cfg):
        self.id = id
        self.cfg = cfg
        self.rollout_manger = get_actor(self.id, "RolloutManager")
        self.data_server = get_actor(self.id, "DataServer")
        self.policy_server = get_actor(self.id, "PolicyServer")
        self.monitor = get_actor(self.id, "Monitor")

        DistributedTrainer = ray.remote(
            **get_resources(cfg.trainer.distributed.resources)
        )(distributed_trainer.DistributedTrainer)

        DataPrefetcher = ray.remote(
            **get_resources(cfg.data_prefetcher.distributed.resources)
        )(data_prefetcher.DataPrefetcher)

        if self.cfg.master_port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("",0))
                self.cfg.master_port = str(s.getsockname()[1])

        self.trainers = [
            DistributedTrainer.options(max_concurrency=10).remote(
                id=default_trainer_id(idx),
                local_rank=idx,
                world_size=self.cfg.num_trainers,  # TODO(jh) check ray resouces if we have so many gpus.
                master_addr=self.cfg.master_addr,
                master_port=self.cfg.master_port,
                master_ifname=self.cfg.get("master_ifname", None),
                gpu_preload=self.cfg.gpu_preload,  # TODO(jh): debug
                local_queue_size=self.cfg.local_queue_size,
                policy_server=self.policy_server,
            )
            for idx in range(self.cfg.num_trainers)
        ]
        
        self.prefetchers = [
            DataPrefetcher.options(max_concurrency=10).remote(
                self.cfg.data_prefetcher, self.trainers, [self.data_server]
            )
            for i in range(self.cfg.num_prefetchers)
        ]

        # cannot start two rollout tasks
        self.semaphore = threading.Semaphore(value=1)
        Logger.info("{} initialized".format(self.id))

        self.eq_dist_list = []

        self.global_steps = 0
        
    def train(self, training_desc: TrainingDesc):
        # create table
        self.table_name = default_table_name(
            training_desc.agent_id,
            training_desc.policy_id,
            training_desc.share_policies,
        )
        ray.get(self.data_server.create_table.remote(self.table_name))

        # save distribution
        self.eq_dist_list.append(training_desc.policy_distributions)
        dump_path = ray.get(self.monitor.get_expr_log_dir.remote())
        dump_path = os.path.join(dump_path, "eq_dist.pkl")
        with open(dump_path, "wb") as f:
            pickle.dump(self.eq_dist_list, f)

        self.rollout_desc = RolloutDesc(
            training_desc.agent_id,
            training_desc.policy_id,
            training_desc.policy_distributions,
            training_desc.share_policies,
            training_desc.sync,
            training_desc.stopper,
        )

        # Setting some trainer cfg through the training description
        self.cfg.trainer.use_intrinsic_reward = training_desc.use_intrinsic_reward
        self.cfg.trainer.use_exploration_bonus = training_desc.use_exploration_bonus

        training_desc.kwargs["cfg"] = self.cfg.trainer
        ray.get([trainer.reset.remote(training_desc) for trainer in self.trainers])

        if self.cfg.gpu_preload:
            ray.get([trainer.local_queue_init.remote() for trainer in self.trainers])
        
        self.training_loop=self.get_traininig_loop()
        
        # start rollout task
        rollout_task_ref = self.rollout_manger.rollout.remote(self.rollout_desc)
        
        # wait for rollout task to completely stop
        global_rollout_steps = ray.get(rollout_task_ref)
        
        # remove table
        ray.get(self.data_server.remove_table.remote(self.table_name))
        
        self.training_loop=None

        return global_rollout_steps

    def train_step(self):
        stopped=next(self.training_loop)
        return stopped
    
    def async_training_loop(self):
        stopped=False
        while not stopped:
            stopped=self.train_step()
        
    def get_traininig_loop(self):
        self.stop_flag = False
        self.stop_flag_lock = threading.Lock()
              
        prefetching_desc = PrefetchingDesc(self.table_name, self.cfg.batch_size)
        prefetching_descs = [prefetching_desc]
        self.prefetching_task_refs = [
            prefetcher.prefetch.remote(prefetching_descs)
            for prefetcher in self.prefetchers
        ]

        training_steps = 0
        # training process
        while True:
            global_timer.record("train_step_start")
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            training_steps += 1
            self.global_steps += 1

            global_timer.record("optimize_start")
            statistics_list = ray.get(
                [trainer.optimize.remote(self.global_steps) for trainer in self.trainers]
            )
            global_timer.time("optimize_start", "optimize_end", "optimize")

            # If we have more than one Distributed Trainer, we can check if the policies are the same
            if len(self.trainers) > 1:
                # Checking to see if the final policies for each distributed trainer is the same    
                # So that we are sure that the distributed training is working correctly
                policy1 = ray.get(self.trainers[0].get_unwrapped_policy.remote())
                policy2 = ray.get(self.trainers[1].get_unwrapped_policy.remote())
               
                target1 = ray.get(self.trainers[0].get_rnd_target.remote())
                target2 = ray.get(self.trainers[1].get_rnd_target.remote())
                
                predictor1 = ray.get(self.trainers[0].get_rnd_predictor.remote())
                predictor2 = ray.get(self.trainers[1].get_rnd_predictor.remote())
                
                for param1, param2 in zip(target1, target2):
                    assert torch.allclose(param1, param2), "Targets are not the same"
                for param1, param2 in zip(predictor1, predictor2):
                    assert torch.allclose(param1, param2), "Predictors are not the same" 
                
                for param1, param2 in zip(policy1.actor.parameters(), policy2.actor.parameters()):
                    assert torch.allclose(param1, param2), "Policies (Actors) are not the same"
                for param1, param2 in zip(policy1.critic.parameters(), policy2.critic.parameters()):
                    assert torch.allclose(param1, param2), "Policies (Critics) are not the same"

            # push new policy
            if training_steps % self.cfg.update_interval == 0:
                global_timer.record("push_policy_start")
                ray.get(self.trainers[0].push_policy.remote(training_steps))
                global_timer.time("push_policy_start", "push_policy_end", "push_policy")

            global_timer.time("train_step_start", "train_step_end", "train_step")

            # reduce
            training_statistics = self.reduce_statistics(
                [statistics[0] for statistics in statistics_list]
            )
            timer_statistics = self.reduce_statistics(
                [statistics[1] for statistics in statistics_list]
            )
            timer_statistics.update(global_timer.elapses)
            data_server_statistics = ray.get(
                self.data_server.get_statistics.remote(self.table_name)
            )

            # log
            main_tag = "Training/{}/{}/".format(
                self.rollout_desc.agent_id, self.rollout_desc.policy_id
            )

            ray.get(
                self.monitor.add_multiple_scalars.remote(
                    main_tag, training_statistics, global_step=training_steps
                )
            )

            # Add plots which only have agent_id as the main tag, this is for having continuous plots throughout the training
            main_tag = "Training/{}/".format(self.rollout_desc.agent_id)

            ray.get(
                self.monitor.add_multiple_scalars.remote(
                    main_tag, training_statistics, global_step=self.global_steps
                )
            )

            main_tag = "TrainingTimer/{}/{}/".format(
                self.rollout_desc.agent_id, self.rollout_desc.policy_id
            )
            ray.get(
                self.monitor.add_multiple_scalars.remote(
                    main_tag, timer_statistics, global_step=training_steps
                )
            )
            main_tag = "DataServer/{}/{}/".format(
                self.rollout_desc.agent_id, self.rollout_desc.policy_id
            )
            ray.get(
                self.monitor.add_multiple_scalars.remote(
                    main_tag, data_server_statistics, global_step=training_steps
                )
            )

            global_timer.clear()
            
            yield False

        # signal prefetchers to stop prefetching
        ray.get(
            [prefetcher.stop_prefetching.remote() for prefetcher in self.prefetchers]
        )
        # wait for prefetching tasks to completely stop
        ray.get(self.prefetching_task_refs)

        Logger.warning("Training ends after {} steps".format(training_steps))
 
        yield True

    def stop_training(self):
        with self.stop_flag_lock:
            self.stop_flag = True

    def reduce_statistics(self, statistics_list):
        statistics = {k: [] for k in statistics_list[0]}
        for s in statistics_list:
            for k, v in s.items():
                statistics[k].append(v)
        for k, v in statistics.items():
            # maybe other reduce method
            statistics[k] = np.mean(v)
        return statistics

    def close(self):
        ray.get([trainer.close.remote() for trainer in self.trainers])
