# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import csv
import argparse
import random
import numpy as np
from statistics import mean, stdev
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import json
from collections import defaultdict

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

def run_game(player_string, action_set, level, render, real_time, seed, thread_idx=None):
    if thread_idx == 0:
        print(f"[THREAD 0] Running game with seed {seed}...")
    else:
        print(f"Running game with seed {seed}...")

    random.seed(seed)
    np.random.seed(seed)
    
    players = player_string.split(';')
    cfg_values = {
        'action_set': action_set,
        'dump_full_episodes': False,
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

def run_games_for_model(model_dir, method_name, games_per_model, num_threads, action_set, level, render, real_time):
    print(f"Evaluating model: {model_dir} from method: {method_name}")
    
    model_results = []
    model_stats = {
        'reward': [], 'goal_diff': [], 'total_pass': [], 'good_pass': [], 'bad_pass': [],
        'total_shot': [], 'good_shot': [], 'bad_shot': [], 'total_possession': [], 'tackle': [],
        'get_tackled': [], 'interception': [], 'get_intercepted': [], 'total_move': []
    }
    
    players = f'football_ai_light:left_players=10,explore=False,checkpoint={model_dir}'
    
    # Calculate how many games to run per thread
    games_per_thread = max(1, games_per_model // num_threads)
    remaining_games = games_per_model
    
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        # Assign unique seed for each game
        seed_start = random.randint(1, 10000)
        thread_idx = 0
        
        while remaining_games > 0:
            batch_size = min(games_per_thread, remaining_games)
            # Each game gets a unique seed in sequence
            futures.append(executor.submit(
                run_games_batch, 
                players, 
                action_set, 
                level, 
                render, 
                real_time, 
                batch_size, 
                seed_start,
                thread_idx  # Pass the thread index
            ))
            
            seed_start += batch_size
            remaining_games -= batch_size
            thread_idx += 1  # Increment thread index
            
        for future in as_completed(futures):
            batch_results, batch_stats = future.result()
            model_results.extend(batch_results)
            for key in model_stats.keys():
                model_stats[key].extend(batch_stats[key])
    
    # Calculate model-level stats
    win_count = sum(1 for result in model_results if result['left'] > result['right'])
    draw_count = sum(1 for result in model_results if result['left'] == result['right'])
    loss_count = sum(1 for result in model_results if result['left'] < result['right'])
    
    # Get all goals scored and conceded
    goals_scored = [result['left'] for result in model_results]
    goals_conceded = [result['right'] for result in model_results]
    
    model_summary = {
        'win_rate': win_count / len(model_results),
        'draw_rate': draw_count / len(model_results),
        'loss_rate': loss_count / len(model_results),
        'goals_scored': mean(goals_scored),
        'goals_conceded': mean(goals_conceded)
    }
    
    # Add average stats
    for key, values in model_stats.items():
        model_summary[key] = mean(values)
        
    return model_summary

def run_games_batch(players, action_set, level, render, real_time, num_games, seed_start, thread_idx=None):
    results = []
    all_stats = {
        'reward': [], 'goal_diff': [], 'total_pass': [], 'good_pass': [], 'bad_pass': [],
        'total_shot': [], 'good_shot': [], 'bad_shot': [], 'total_possession': [], 'tackle': [],
        'get_tackled': [], 'interception': [], 'get_intercepted': [], 'total_move': []
    }
    
    for i in range(num_games):
        current_seed = seed_start + i
        if thread_idx == 0:
            print(f"[THREAD 0] Running game {i+1}/{num_games} with seed {current_seed}...")
        else:
            print(f"Running game {i+1}/{num_games} with seed {current_seed}...")
        
        score, stats = run_game(players, action_set, level, render, real_time, current_seed, thread_idx)
        results.append(score)
        for key in all_stats.keys():
            all_stats[key].append(stats[key])
    
    return results, all_stats

def eval_methods(methods_config, games_per_model, num_threads, output_dir):
    action_set = 'full'
    level = 'full_game_10_vs_10_challenge'
    render = False
    real_time = False
    
    os.makedirs(output_dir, exist_ok=True)
    
    methods_results = {}
    
    for method_name, model_dirs in methods_config.items():
        print(f"Evaluating method: {method_name} with {len(model_dirs)} models")
        methods_results[method_name] = []
        
        # Evaluate each model in the method
        for model_dir in model_dirs:
            model_summary = run_games_for_model(
                model_dir, 
                method_name, 
                games_per_model, 
                num_threads, 
                action_set, 
                level, 
                render, 
                real_time
            )
            
            methods_results[method_name].append(model_summary)
        
        # Write individual model results for this method
        write_model_results(method_name, methods_results[method_name], output_dir)
    
    # Calculate and write method-level aggregated results
    write_method_results(methods_results, output_dir)
    
    print("Evaluation complete.")

def write_model_results(method_name, model_results, output_dir):
    """Write detailed results for each model in a method"""
    output_file = os.path.join(output_dir, f"{method_name}_models.csv")
    
    if not model_results:
        return
    
    # Get field names from the first model result
    fieldnames = list(model_results[0].keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model_result in model_results:
            writer.writerow(model_result)

def write_method_results(methods_results, output_dir):
    """Write aggregated results for each method"""
    output_file = os.path.join(output_dir, "method_comparison.csv")
    
    # Calculate averages and std deviations for each method
    method_summaries = {}

    for method_name, model_results in methods_results.items():
        if not model_results:
            continue
            
        method_summary = defaultdict(float)
        all_model_values = defaultdict(list)
        
        # Collect values for each metric across all models
        for model_result in model_results:
            for field, value in model_result.items():
                all_model_values[field].append(value)
        
        # Calculate average across models for each metric
        for field, values in all_model_values.items():
            method_summary[f'avg_{field}'] = mean(values)
            method_summary[f'std_{field}'] = stdev(values) if len(values) > 1 else 0.0
            
        method_summaries[method_name] = dict(method_summary)
    
    # Write to CSV
    if not method_summaries:
        return
        
    # Get field names from the first method summary
    first_method = next(iter(method_summaries.values()))
    fieldnames = ["method"] + list(first_method.keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, summary in method_summaries.items():
            row = {'method': method_name}
            row.update(summary)
            writer.writerow(row)

def update_method_results(method_name, model_results, output_dir):
    # This function takes the existing {method_name}_models.csv file in the output_dir and concatenates the new results to it

    output_file = os.path.join(output_dir, f"{method_name}_models.csv")

    if not model_results:
        print("No model results to update")
        return
    
    # Concatenate the new results as a new row in the existing csv file for this method
    with open(output_file, 'a', newline='') as csvfile:

        fieldnames = list(model_results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for model_result in model_results:
            writer.writerow(model_result)

def write_method_results_from_individual_files(output_dir):
    # This method takes the individual method csv files, i.e., {method_name}_models.csv, in the output_dir and creates the method_comparison.csv file

    method_summaries = {}

    for file in os.listdir(output_dir):
        if file.endswith("_models.csv"):
            method_name = file.replace("_models.csv", "")
            method_summaries[method_name] = []

            with open(os.path.join(output_dir, file), 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    method_summaries[method_name].append(row)

    # Calculate averages and std deviations for each method
    method_summaries_aggregated = {}

    for method_name, model_results in method_summaries.items():
        
        method_summary = defaultdict(float)
        all_model_values = defaultdict(list)
        
        # Collect values for each metric across all models
        for model_result in model_results:
            for field, value in model_result.items():
                all_model_values[field].append(float(value))
        
        # Calculate average across models for each metric
        for field, values in all_model_values.items():
            current_arr = np.array(values)
            q25, q75 = np.percentile(current_arr, [25, 75]) 
            iq = current_arr[(current_arr >= q25) & (current_arr <= q75)] 
            if q25 not in iq:
                iq = np.insert(iq, 0, q25)
            if q75 not in iq:
                iq = np.append(iq, q75)
            iqm_val = round(np.mean(iq), 2)
            iqm_std_val = round(np.std(iq), 2)

            # Arithmetic mean and standard deviation 
            # method_summary[f'avg_{field}'] = mean(values)
            # method_summary[f'std_{field}'] = stdev(values) if len(values) > 1 else 0.0

            # Interquartile mean and standard deviation 
            method_summary[f'avg_{field}'] = iqm_val
            method_summary[f'std_{field}'] = iqm_std_val
            
        method_summaries_aggregated[method_name] = dict(method_summary)

    # Write to CSV
    if not method_summaries_aggregated:
        return
    
    output_file = os.path.join(output_dir, "method_comparison.csv")
    # Get field names from the first method summary
    first_method = next(iter(method_summaries_aggregated.values()))
    fieldnames = ["method"] + list(first_method.keys())

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, summary in method_summaries_aggregated.items():
            row = {'method': method_name}
            row.update(summary)
            writer.writerow(row)

def update_evals(methods_config, games_per_model, num_threads, output_dir):

    action_set = 'full'
    level = 'full_game_10_vs_10_challenge'
    render = False
    real_time = False

    os.makedirs(output_dir, exist_ok=True)

    methods_results = {}

    for method_name, model_dirs in methods_config.items():
        print(f"Evaluating method: {method_name} with {len(model_dirs)} models")
        methods_results[method_name] = []

        # Evaluate each model in the method
        for model_dir in model_dirs:
            model_summary = run_games_for_model(
                model_dir,
                method_name,
                games_per_model,
                num_threads,
                action_set,
                level,
                render,
                real_time
            )

            methods_results[method_name].append(model_summary)

        # Write individual model results for this method
        update_method_results(method_name, methods_results[method_name], output_dir)

    # Calculate and write method-level aggregated results
    write_method_results_from_individual_files(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='JSON config file with methods and their model directories')
    parser.add_argument('--games', type=int, default=250, 
                        help='Number of games to play per model')
    parser.add_argument('--threads', type=int, default=10, 
                        help='Number of threads to use')
    parser.add_argument('--output', type=str, default='results', 
                        help='Output directory for results')
    parser.add_argument('--update_evaluations', action='store_true', default=False)
    parser.add_argument('--aggregate', action='store_true', default=False)

    args = parser.parse_args()

    # Read config file
    with open(args.config, 'r') as f:
        methods_config = json.load(f)
    
    start_time = time.time()
    if args.aggregate:
        write_method_results_from_individual_files(args.output)
    elif args.update_evaluations:
        update_evals(methods_config, args.games, args.threads, args.output)
    else:
        eval_methods(methods_config, args.games, args.threads, args.output)
    end_time = time.time()
    print(f'Evaluation took {end_time - start_time} seconds')