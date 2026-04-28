# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.lines import Line2D

# Set the style for seaborn and adjust font sizes.
sns.set_theme(context='paper', style="whitegrid")
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Function to smooth the data.
def smooth_data(data, weight=0.3):
    ema = [data[0]]
    for point in data[1:]:
        ema_current = weight * point + (1 - weight) * ema[-1]
        ema.append(ema_current)
    return ema

# Function to load an event file and extract scalar data.
def load_event_file(event_file):
    event_acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    data = {tag: [] for tag in tags}
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data

# Function to load experiment data from a directory.
# It finds the first event file inside the directory (searching recursively).
def load_experiment_data(exp_dir):
    event_files = glob.glob(os.path.join(exp_dir, '**', 'event*'), recursive=True)
    if not event_files:
        print(f"No event files found in {exp_dir}")
        return {}
    # Load the first event file found.
    print(f"Loading event file {event_files[0]} from {exp_dir}")
    return load_event_file(event_files[0])

# Modified function to plot combined metrics.
# Assumes all experiments have the same number of steps.
# The first experiment is drawn in red, the second in blue.
# Only the solid (smoothed) lines are included in the legend using custom names,
# and a vertical dashed purple line is drawn (and added to the legend) to indicate Intrinsic Reward Used.
def plot_combined_metrics(experiments_data, metric_name, intrinsic_used_after,
                          legend_names, file_name='combined_plot', xlabel='Steps', ylabel=None, title=None):
    
    plt.figure(figsize=(10, 6))
    
    # Fixed colors: first experiment red, second blue.
    fixed_colors = ['green', 'dodgerblue']
    legend_elements = []
    
    # Iterate over experiments in the order of the provided dictionary.
    z_value = 2
    for i, (exp_key, data) in enumerate(experiments_data.items()):
        color = fixed_colors[i] if i < len(fixed_colors) else 'black'
        # Extract steps and values (assumes each experiment's metric has the same length)
        steps, values = zip(*data[metric_name])
        smoothed_values = smooth_data(values, weight=0.1)
        
        # Plot the raw values with shading (not added to legend)
        plt.plot(steps, values, color=color, alpha=0.2, zorder=z_value)
        # Plot the solid (smoothed) line (to be shown in legend)
        plt.plot(steps, smoothed_values, color=color, linewidth=1, zorder=z_value)
        
        # Use the provided custom legend name.
        custom_label = legend_names.get(exp_key, exp_key)
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=custom_label))

        z_value -= 1
    
    # Draw a vertical dashed purple line at the intrinsic_used_after value.
    plt.axvline(x=intrinsic_used_after, color='black', linestyle='--', linewidth=1.6, zorder=3)
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=2,
                                  label='RND Warm-up Ended'))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else metric_name)
    plt.title(title if title else f'Combined {metric_name} Over Steps')
    
    # Add the legend (only with the solid lines and intrinsic reward line).
    plt.legend(handles=legend_elements)
    plt.axhline(0, color='black', linewidth=1.5)
    
    # Save the plot to the ./plots directory.
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/{file_name}.png')
    plt.close()

if __name__ == '__main__':
    # List of experiment directories.
    experiments_dirs = ["./rnd/2025-02-05-02-57-09", "./vanilla-pid-mlp/2025-07-08-07-10-44"]
    
    # Build a dictionary mapping experiment name (from directory basename) to loaded data.
    experiments_data = {}
    for exp_dir in experiments_dirs:
        exp_name = os.path.basename(os.path.normpath(exp_dir))
        data = load_experiment_data(exp_dir)
        if data:  # Only add if data was successfully loaded.
            experiments_data[exp_name] = data

    # Calculate the environment step when intrinsic reward and RND is used.
    after_ppo_epochs = 700
    num_workers = 40
    num_env_steps_per_worker = 500
    intrinsic_used_after = after_ppo_epochs * num_workers * num_env_steps_per_worker

    # Define custom legend names for each experiment.
    custom_legend_names = {
        "2025-02-05-02-57-09": "TiZero-RND",
        "2025-07-08-07-10-44": "TiZero"
    }

    # Plot the "win" metric.
    plot_combined_metrics(
        experiments_data=experiments_data,
        metric_name="Rollout/agent_0/win",
        intrinsic_used_after=intrinsic_used_after,
        legend_names=custom_legend_names,
        file_name='combined_win',
        xlabel='Env Steps',
        ylabel='Win-Rate',
        title='Win-Rate Over Environment Steps'
    )
    
    # Plot the "reward" metric.
    plot_combined_metrics(
        experiments_data=experiments_data,
        metric_name="Rollout/agent_0/reward",
        intrinsic_used_after=intrinsic_used_after,
        legend_names=custom_legend_names,
        file_name='combined_reward',
        xlabel='Env Steps',
        ylabel='Reward',
        title='Reward Over Environment Steps'
    )
    
    print("Plots generated in the ./plots directory.")

