# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from light_malib.utils.logger import Logger
import ray
import argparse
from light_malib.utils.cfg import load_cfg, convert_to_easydict
from light_malib.utils.random import set_random_seed
from light_malib.framework.tizero_runner import TiZeroRunner
import time
import sys
import os
import yaml
from omegaconf import OmegaConf

import pathlib

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.append(BASE_DIR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0) # given global seed
    parser.add_argument("--rollout-seed", type=int, default=100) # given rollout manager seed
    args = parser.parse_args()
    return args


def get_local_ip_address():
    import socket

    ip_address = socket.gethostbyname(socket.gethostname())

    return ip_address


def start_cluster():
    try:
        cluster_start_info = ray.init(address="local")
    except ConnectionError:
        Logger.warning("No active cluster detected, will create local ray instance.")
        cluster_start_info = ray.init(resources={})

    Logger.warning(
        "============== Cluster Info ==============\n{}".format(cluster_start_info)
    )
    Logger.warning("* cluster resources:\n{}".format(ray.cluster_resources()))
    Logger.warning(
        "this worker ip: {}".format(ray.get_runtime_context().worker.node_ip_address)
    )
    return cluster_start_info


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # Set the general and the rollout manager seed to the given seeds
    print("Setting global seed to {}".format(args.seed))
    print("Setting rollout seed to {}".format(args.rollout_seed))
    set_random_seed(args.seed)
    cfg.rollout_manager.seed = args.rollout_seed

    assert cfg.distributed.nodes.master.ip is not None
    cluster_start_info = start_cluster()
    print("Started cluster")

    if cfg.distributed.nodes.master.ip == "auto":
        ip = ray.get_runtime_context().worker.node_ip_address
        cfg.distributed.nodes.master.ip = ip
        print("Automatically set master ip to local ip address: {}".format(ip))
        Logger.warning("Automatically set master ip to local ip address: {}".format(ip))

    # check cfg
    # check gpu number here
    assert (
        cfg.training_manager.num_trainers <= ray.cluster_resources()["GPU"]
    ), "#trainers({}) should be <= #gpus({})".format(
        cfg.training_manager.num_trainers, ray.cluster_resources()["GPU"]
    )
    # check batch size here
    assert (
        cfg.training_manager.batch_size <= cfg.data_server.table_cfg.capacity
    ), "batch_size({}) should be <= capacity({})".format(
        cfg.training_manager.batch_size, cfg.data_server.table_cfg.capacity
    )
    # check sync_training
    if cfg.framework.sync_training and cfg.framework.get('on_policy', True):
        assert cfg.data_server.table_cfg.sample_max_usage==1
        assert cfg.training_manager.batch_size==cfg.rollout_manager.batch_size
        assert cfg.rollout_manager.worker.sample_length<=0

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cfg.expr_log_dir = os.path.join(
        cfg.log_dir, cfg.expr_group, cfg.expr_name, timestamp
    )
    # Get the home directory of the current user and join it with the expr_log_dir
    cfg.expr_log_dir = os.path.join(os.path.expanduser("~"), cfg.expr_log_dir)
    # cfg.expr_log_dir = os.path.join(BASE_DIR, cfg.expr_log_dir)
    os.makedirs(cfg.expr_log_dir, exist_ok=True)

    # copy config file
    yaml_path = os.path.join(cfg.expr_log_dir, "config.yaml")
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_ed = convert_to_easydict(cfg)

    from light_malib.monitor.monitor import Monitor
    from light_malib.utils.distributed import get_resources

    # Add Monitor
    Monitor = ray.remote(**get_resources(cfg_ed.monitor.distributed.resources))(Monitor)
    monitor = Monitor.options(name="Monitor", max_concurrency=5).remote(cfg_ed)

    # Initialize the runner
    runner = TiZeroRunner(cfg_ed, cfg)

    try:
        runner.run()
    except KeyboardInterrupt as e:
        Logger.warning(
            "Detected KeyboardInterrupt event, start background resources recycling threads ..."
        )
    finally:
        runner.close()
        ray.get(monitor.close.remote())
        ray.shutdown()


if __name__ == "__main__":
    main()
