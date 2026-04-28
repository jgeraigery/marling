# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import csv
import argparse
import random
import numpy as np
from statistics import mean
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
import time

from gfootball.env import football_env, config
from light_malib.envs.gr_football.state import State
from light_malib.envs.gr_football.stats_basic import StatsCaculator
from light_malib.envs.gr_football.rewarder_basic import Rewarder
from light_malib.envs.gr_football.tools.action_set import *
from gfootball.env.football_action_set import CoreAction

def get_reward(states, rewarder, rewards):
    rewarder.reset_reward_state()
    rewards = [rewards] * 10
    rewards = [
        [rewarder.calc_reward(reward, state, states, team=0)]
        for idx, (reward, state) in enumerate(zip(rewards, states))
    ]
    return rewards

def run_game(player_string, action_set, level, render, real_time, seed):
    print(f"Running game with seed {seed}...")

    random.seed(seed)
    np.random.seed(seed)
    
    players = player_string.split(';')
    cfg_values = {
        'action_set': action_set,
        'dump_full_episodes': True,
        'players': players,
        'real_time': real_time,
    }
    if level:
        cfg_values['level'] = level
    cfg = config.Config(cfg_values)
    env = football_env.FootballEnv(cfg)
    if render:
        env.render()
    obs = env.reset()

    reward_config = {
        "out_of_bound_penalty": 0.001,
        "gather_penalty": 0.001,
        "pass_reward": 0.05,
        "hold_ball": 0.1,
        "win_reward": None,
        "yellow_reward": None,
        "min_dist_reward": None,
        "total_dist_to_ball_reward": 0,
        "different_action_reward": 0,
        "goal_reward": None,
        "official_reward": 1,  
    }
    states = [State(n_player=10) for i in range(10)]
    rewarder = Rewarder(reward_config, num_players_per_team=10)
    
    stats_calculator = StatsCaculator()
    stats_calculator.reset()
   
    obs = env._convert_observations(obs, env._players[0], 0, 0) 
    for o, s in zip(obs, states):
        s.update_obs(o)

    score = {'left': 0, 'right': 0}
    try:
        while True:
            actions = env._get_actions()
            obs, rew, done, info = env.step([])
         
            obs = env._convert_observations(obs, env._players[0], 0, 0)
            for o, a, s in zip(obs, actions, states):
                s.update_obs(o)
                s.update_action(a)
           
            rewards = get_reward(states, rewarder, rew)
            
            for idx, state in enumerate(states):
                stats_calculator.calc_stats(state, rewards[idx][0], idx)
             
            if rew:
                if rew > 0:
                    score['left'] += rew
                else:
                    score['right'] -= rew
            if done:
                break
        return score, stats_calculator.stats
    except KeyboardInterrupt:
        print('Game stopped, writing dump...')
        env.write_dump('shutdown')

def eval_model(dir, num_games, num_threads, output):
    players = f'football_ai_light:left_players=10,explore=True,checkpoint={dir}'
    action_set = 'full'
    level = 'full_game_10_vs_10_challenge'
    render = False
    real_time = False

    all_stats = {
        'reward': [], 'goal_diff': [], 'total_pass': [], 'good_pass': [], 'bad_pass': [],
        'total_shot': [], 'good_shot': [], 'bad_shot': [], 'total_possession': [], 'tackle': [],
        'get_tackled': [], 'interception': [], 'get_intercepted': [], 'total_move': []
    }

    results = []

    print(f'Starting evaluation: {num_games} games per seed across 10 seeds with {num_threads} threads.')

    # Loop over 10 seeds, processing in batches of 5 threads
    seeds = range(1, 11)
    for seed_batch_start in range(0, len(seeds), num_threads):
        seed_batch = seeds[seed_batch_start:seed_batch_start + num_threads]
        print(f'Starting seed batch: {seed_batch}')
        
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for seed in seed_batch:
                futures.append(executor.submit(run_games_batch, players, action_set, level, render, real_time, num_games, seed))

            for future in as_completed(futures):
                batch_results, batch_stats = future.result()
                results.extend(batch_results)
                for key in all_stats.keys():
                    all_stats[key].extend(batch_stats[key])

        print(f'Completed seed batch: {seed_batch}')

    print(f"{len(results)} game were run. Calculating stats...")
    win_count = sum(1 for result in results if result['left'] > result['right'])
    draw_count = sum(1 for result in results if result['left'] == result['right'])
    loss_count = sum(1 for result in results if result['left'] < result['right'])
    avg_goals_scored = mean(result['left'] for result in results)
    avg_goals_conceded = mean(result['right'] for result in results)

    avg_stats = {key: mean(values) for key, values in all_stats.items()}

    print("Writing results to CSV...")
    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['win_rate', 'draw_rate', 'loss_rate', 'avg_goals_scored', 'avg_goals_conceded'] + list(avg_stats.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'win_rate': win_count / len(results),
            'draw_rate': draw_count / len(results),
            'loss_rate': loss_count / len(results),
            'avg_goals_scored': avg_goals_scored,
            'avg_goals_conceded': avg_goals_conceded,
            **avg_stats
        })

    print("Evaluation complete.")

def run_games_batch(players, action_set, level, render, real_time, num_games, seed):
    results = []
    all_stats = {
        'reward': [], 'goal_diff': [], 'total_pass': [], 'good_pass': [], 'bad_pass': [],
        'total_shot': [], 'good_shot': [], 'bad_shot': [], 'total_possession': [], 'tackle': [],
        'get_tackled': [], 'interception': [], 'get_intercepted': [], 'total_move': []
    }
    
    for i in range(num_games):
        print(f"Running game {i+1}/{num_games} for seed {seed}...")
        score, stats = run_game(players, action_set, level, render, real_time, seed)
        results.append(score)
        for key in all_stats.keys():
            all_stats[key].append(stats[key])
    
    return results, all_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to the model checkpoints')
    parser.add_argument('--n', type=int, default=10, help='Number of games to play per seed')
    parser.add_argument('--threads', type=int, default=5, help='Number of threads to use')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file')

    args = parser.parse_args()

    start_time = time.time()
    eval_model(args.dir, args.n, args.threads, args.output)
    end_time = time.time()
    print(f'Evaluation took {end_time - start_time} seconds')
