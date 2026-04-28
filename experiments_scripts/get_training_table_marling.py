# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import os
import glob
import csv
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Methods to generate the metrics for (method name, faulty event file with the last faulty index indicated to generate correct results)
METHODS = [("vanilla-pid-mlp", {}),
        #    ("ssir", {}), 
           ("rnd-pid-mlp", {})]

# CSV Headers for metrics with interquartile mean and standard deviation 
COLUMNS = [
    "Methods",
    "IQM Number of Diffs Passed", "Std. Number of Diffs Passed",
    "IQM Number of Rollout Steps per Diff", "Std. Rollout Steps per Diff",
    "IQM Elapsed Time per Diff", "Std. Elapsed Time per Diff",
    "IQM Challenge Phases Passed", "Std. Challenge Phases Passed",
    "IQM Steps per Challenge Phase", "Std. Steps per Challenge Phase",
    "IQM Elapsed Time per Challenge Phase", "Std. Elapsed Time per Challenge Phase",
    "IQM Generalize Phases Passed", "Std. Generalize Phases Passed",
    "IQM Steps per Generalize Phase", "Std. Steps per Generalize Phase",
    "IQM Elapsed Time per Generalize Phase", "Std. Elapsed Time per Generalize Phase"
]

def get_event_files(method_name):
    pattern = f"./{method_name}/**/event*"
    return glob.glob(pattern, recursive=True)

def compute_number_of_diffs_passed(ea):
    """Compute the number of difficulties passed for a single experiment"""
    # Check if agent completed all 10 difficulties (index 11 means all 10 were passed)
    if "Training/agent_0/agent_0-default-11/V_mean" in ea.scalars.Keys():
        return 10
    else:
        # Find the highest difficulty idx
        max_diff = 0
        for tag in ea.scalars.Keys():
            if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
                # Extract difficulty index
                parts = tag.split("/")
                agent_part = parts[2]  # "agent_0-default-{idx}"
                diff_idx = int(agent_part.split("-")[-1])
                max_diff = max(max_diff, diff_idx)
        
        return max_diff - 1

def compute_avg_rollout_steps_per_diff(ea):
    """Compute the avg number of rollout steps per diff for a single experiment"""
    total_steps = 0
    diff_count = 0
    
    # Find all difficulty tags and get max steps for each
    for i in range(1, 11):  # Check difficulties 1 through 10
        tag = f"Training/agent_0/agent_0-default-{i}/V_mean"
        next_tag = f"Training/agent_0/agent_0-default-{i + 1}/V_mean"
        if tag in ea.scalars.Keys() and next_tag in ea.scalars.Keys():
            # Get all events for this tag
            events = ea.Scalars(tag)
            if events:
                # Find max step (last x-axis value)
                max_step = events[-1].step
                total_steps += max_step
                diff_count += 1
    
    # Calculate average steps per difficulty
    if diff_count > 0:
        return total_steps / diff_count
    else:
        return 0

def compute_avg_elapsed_time_per_diff(ea):
    """Compute the avg elapsed time per diff for a single experiment"""
    total_time = 0
    diff_count = 0
    
    # Find all difficulty tags and get elapsed time for each
    for i in range(1, 11):  # Check difficulties 1 through 10
        tag = f"Training/agent_0/agent_0-default-{i}/V_mean"
        next_tag = f"Training/agent_0/agent_0-default-{i + 1}/V_mean"
        if tag in ea.scalars.Keys() and next_tag in ea.scalars.Keys():
            # Get all events for this tag
            events = ea.Scalars(tag)
            if events:
                # Find the last event and get its wall time
                last_event = events[-1]
                first_event = events[0]
                
                # Calculate elapsed time in seconds
                elapsed_time = (last_event.wall_time - first_event.wall_time) / 3600.0
                total_time += elapsed_time
                diff_count += 1
    
    # Calculate average elapsed time per difficulty
    if diff_count > 0:
        return total_time / diff_count
    else:
        return 0

def compute_challenge_phases_passed(ea, last_faulty_idx = None):
    """Compute the number of challenge phases passed for a single experiment"""
    challenge_indices = []
    
    # Find all challenge phase tags (indices 11, 13, 15, etc.)
    for tag in ea.scalars.Keys():
        if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
            # Extract difficulty index
            parts = tag.split("/")
            agent_part = parts[2]  # "agent_0-default-{idx}"
            diff_idx = int(agent_part.split("-")[-1])
            
            # Check if it's a challenge phase (odd numbers starting from 11)
            next_tag = f"Training/agent_0/agent_0-default-{diff_idx + 1}/V_mean"
            if diff_idx >= 11 and diff_idx % 2 == 1:
                if last_faulty_idx != None and (diff_idx + 1) == last_faulty_idx:
                    continue
                elif next_tag in ea.scalars.Keys():
                    challenge_indices.append(diff_idx)
                    
    # The number of completed challenge phases is the count - 1 (excluding the last one)
    if len(challenge_indices) > 0:
        return len(challenge_indices)
    else:
        return 0

def compute_avg_rollout_steps_per_challenge_phase(ea, last_faulty_idx=None):
    """Compute the avg number of steps per challenge phase for a single experiment"""
    total_steps = 0
    challenge_indices = []
    
    # Find all challenge phase tags (indices 11, 13, 15, etc.)
    for tag in ea.scalars.Keys():
        if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
            # Extract difficulty index
            parts = tag.split("/")
            agent_part = parts[2]  # "agent_0-default-{idx}"
            diff_idx = int(agent_part.split("-")[-1])
            
            # Check if it's a challenge phase (odd numbers starting from 11)
            next_tag = f"Training/agent_0/agent_0-default-{diff_idx + 1}/V_mean"
            if diff_idx >= 11 and diff_idx % 2 == 1:
                if last_faulty_idx != None and (diff_idx + 1) == last_faulty_idx:
                    continue
                elif next_tag in ea.scalars.Keys():
                    challenge_indices.append(diff_idx)

    # Calculate total steps for all challenge phases (excluding the last one)
    for idx in challenge_indices:
        tag = f"Training/agent_0/agent_0-default-{idx}/V_mean"
        if tag in ea.scalars.Keys():
            events = ea.Scalars(tag)
            if events:
                # Find max step (last x-axis value)
                max_step = events[-1].step
                total_steps += max_step
    
    # Calculate average steps per challenge phase
    if len(challenge_indices) > 0:
        return total_steps / len(challenge_indices)
    else:
        return 0

def compute_avg_elapsed_time_per_challenge_phase(ea, last_faulty_idx=None):
    """Compute the avg elapsed time per challenge phase for a single experiment"""
    total_time = 0
    challenge_indices = []
    
    # Find all challenge phase tags (indices 11, 13, 15, etc.)
    for tag in ea.scalars.Keys():
        if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
            # Extract difficulty index
            parts = tag.split("/")
            agent_part = parts[2]  # "agent_0-default-{idx}"
            diff_idx = int(agent_part.split("-")[-1])
            
            # Check if it's a challenge phase (odd numbers starting from 11)
            next_tag = f"Training/agent_0/agent_0-default-{diff_idx + 1}/V_mean"
            if diff_idx >= 11 and diff_idx % 2 == 1:
                if last_faulty_idx != None and (diff_idx + 1) == last_faulty_idx:
                    continue
                elif next_tag in ea.scalars.Keys():
                    challenge_indices.append(diff_idx)
    
    # Calculate total elapsed time for all challenge phases (excluding the last one)
    for idx in challenge_indices:
        tag = f"Training/agent_0/agent_0-default-{idx}/V_mean"
        if tag in ea.scalars.Keys():
            events = ea.Scalars(tag)
            if events:
                # Calculate elapsed time between first and last event
                first_event = events[0]
                last_event = events[-1]
                elapsed_time = (last_event.wall_time - first_event.wall_time) / 3600.0
                total_time += elapsed_time
    
    # Calculate average elapsed time per challenge phase
    if len(challenge_indices) > 0:
        return total_time / len(challenge_indices)
    else:
        return 0

def compute_generalize_phases_passed(ea, last_faulty_idx=None):
    """Compute the number of generalize phases passed for a single experiment"""
    generalize_indices = []
    
    # Find all generalize phase tags (indices 12, 14, 16, etc.)
    for tag in ea.scalars.Keys():
        if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
            # Extract difficulty index
            parts = tag.split("/")
            agent_part = parts[2]  # "agent_0-default-{idx}"
            diff_idx = int(agent_part.split("-")[-1])
            
            # Check if it's a generalize phase (even numbers starting from 12)
            next_tag = f"Training/agent_0/agent_0-default-{diff_idx + 1}/V_mean"
            if diff_idx >= 12 and diff_idx % 2 == 0:
                if last_faulty_idx != None and (diff_idx + 1) == last_faulty_idx:
                    continue
                elif next_tag in ea.scalars.Keys():
                    generalize_indices.append(diff_idx)
    
    # The number of completed generalize phases is the count - 1 (excluding the last one)
    if len(generalize_indices) > 0:
        return len(generalize_indices)
    else:
        return 0

def compute_avg_rollout_steps_per_generalize_phase(ea, last_faulty_idx=None):
    """Compute the avg number of steps per generalize phase for a single experiment"""
    total_steps = 0
    generalize_indices = []
    
    # Find all generalize phase tags (indices 12, 14, 16, etc.)
    for tag in ea.scalars.Keys():
        if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
            # Extract difficulty index
            parts = tag.split("/")
            agent_part = parts[2]  # "agent_0-default-{idx}"
            diff_idx = int(agent_part.split("-")[-1])
            
            # Check if it's a generalize phase (even numbers starting from 12)
            next_tag = f"Training/agent_0/agent_0-default-{diff_idx + 1}/V_mean"
            if diff_idx >= 12 and diff_idx % 2 == 0:
                if last_faulty_idx != None and (diff_idx + 1) == last_faulty_idx:
                    continue
                elif next_tag in ea.scalars.Keys():
                    generalize_indices.append(diff_idx)
    
    # Calculate total steps for all generalize phases (excluding the last one)
    for idx in generalize_indices:
        tag = f"Training/agent_0/agent_0-default-{idx}/V_mean"
        if tag in ea.scalars.Keys():
            events = ea.Scalars(tag)
            if events:
                # Find max step (last x-axis value)
                max_step = events[-1].step
                total_steps += max_step
    
    # Calculate average steps per generalize phase
    if len(generalize_indices) > 0:
        return total_steps / len(generalize_indices)
    else:
        return 0

def compute_avg_elapsed_time_per_generalize_phase(ea, last_faulty_idx=None):
    """Compute the avg elapsed time per generalize phase for a single experiment in hours"""
    total_time = 0
    generalize_indices = []
    
    # Find all generalize phase tags (indices 12, 14, 16, etc.)
    for tag in ea.scalars.Keys():
        if tag.startswith("Training/agent_0/agent_0-default-") and tag.endswith("/V_mean"):
            # Extract difficulty index
            parts = tag.split("/")
            agent_part = parts[2]  # "agent_0-default-{idx}"
            diff_idx = int(agent_part.split("-")[-1])
            
            # Check if it's a generalize phase (even numbers starting from 12)
            next_tag = f"Training/agent_0/agent_0-default-{diff_idx + 1}/V_mean"
            if diff_idx >= 12 and diff_idx % 2 == 0:
                if last_faulty_idx != None and (diff_idx + 1) == last_faulty_idx:
                    continue
                elif next_tag in ea.scalars.Keys():
                    generalize_indices.append(diff_idx)
    
    # Calculate total elapsed time for all generalize phases (excluding the last one)
    for idx in generalize_indices:
        tag = f"Training/agent_0/agent_0-default-{idx}/V_mean"
        if tag in ea.scalars.Keys():
            events = ea.Scalars(tag)
            if events:
                # Calculate elapsed time between first and last event (in hours)
                first_event = events[0]
                last_event = events[-1]
                elapsed_time = (last_event.wall_time - first_event.wall_time) / 3600.0  # Convert to hours
                total_time += elapsed_time
    
    # Calculate average elapsed time per generalize phase
    if len(generalize_indices) > 0:
        return total_time / len(generalize_indices)
    else:
        return 0

def compute_exp_stats(event_file, results, last_faulty_idx=None):
    """Load event file once and compute all metrics for a single experiment"""
    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Calculate all metrics
    results["diffs_passed"].append(compute_number_of_diffs_passed(ea))
    results["rollout_steps_per_diff"].append(compute_avg_rollout_steps_per_diff(ea))
    results["elapsed_time_per_diff"].append(compute_avg_elapsed_time_per_diff(ea))
    num_challenge_phases_passed = compute_challenge_phases_passed(ea, last_faulty_idx=last_faulty_idx)
    results["challenge_phases_passed"].append(num_challenge_phases_passed)
    if num_challenge_phases_passed > 0:
        results["steps_per_challenge_phase"].append(compute_avg_rollout_steps_per_challenge_phase(ea, last_faulty_idx=last_faulty_idx))
        results["elapsed_time_per_challenge_phase"].append(compute_avg_elapsed_time_per_challenge_phase(ea, last_faulty_idx=last_faulty_idx))
    num_generalize_phases_passed = compute_generalize_phases_passed(ea, last_faulty_idx=last_faulty_idx)
    results["generalize_phases_passed"].append(num_generalize_phases_passed)
    if num_generalize_phases_passed > 0:
        results["steps_per_generalize_phase"].append(compute_avg_rollout_steps_per_generalize_phase(ea, last_faulty_idx=last_faulty_idx))
        results["elapsed_time_per_generalize_phase"].append(compute_avg_elapsed_time_per_generalize_phase(ea, last_faulty_idx=last_faulty_idx))

def export_method_training_stats(method_name, faulty_event_files = None):

    event_files = get_event_files(method_name)
    
    if not event_files:
        print(f"No event files found for method {method_name}")
        return [method_name] + [0] * (len(COLUMNS) - 1)
    
    results = {
        "diffs_passed": [],
        "rollout_steps_per_diff": [],
        "elapsed_time_per_diff": [],
        "challenge_phases_passed": [],
        "steps_per_challenge_phase": [],
        "elapsed_time_per_challenge_phase": [],
        "generalize_phases_passed": [],
        "steps_per_generalize_phase": [],
        "elapsed_time_per_generalize_phase": []
    }
    
    for event_file in event_files:
        print(f"Processing {event_file}")
        if faulty_event_files != None and event_file in faulty_event_files.keys():
            print(f"This event file has a faulty last idx of {faulty_event_files[event_file]}")
            compute_exp_stats(event_file, results, last_faulty_idx=faulty_event_files[event_file])
        else:
            compute_exp_stats(event_file, results)

    row = [method_name]
    for key in results:
        if results[key]:

            current_arr = np.array(results[key])

            q25, q75 = np.percentile(current_arr, [25, 75]) 
            iq = current_arr[(current_arr >= q25) & (current_arr <= q75)] 

            if q25 not in iq:
                iq = np.insert(iq, 0, q25)
            if q75 not in iq:
                iq = np.append(iq, q75)

            iqm_val = round(np.mean(iq), 2)
            iqm_std_val = round(np.std(iq), 2)

        else:
            iqm_val = 0.00 
            iqm_std_val = 0.00 

        row.append(iqm_val)
        row.append(iqm_std_val)
    
    return row

def main():
    all_results = []
    for method, faulty_event_files in METHODS:
        method_results = export_method_training_stats(method, faulty_event_files=faulty_event_files)
        all_results.append(method_results)
    
    with open('method_comparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COLUMNS)

        # format numeric values with higher floating-point precision for CSV output
        formatted_rows = []
        for row in all_results:
            formatted_row = []
            for v in row:
                try:
                    # try to convert to float and format; leaves non-numeric (e.g. method name) untouched
                    fv = float(v)
                    formatted_row.append(f"{fv:.6f}")  # use .6f or adjust precision as needed
                except Exception:
                    formatted_row.append(v)
            formatted_rows.append(formatted_row)

        writer.writerows(formatted_rows)
    
    print(f"Results written to method_comparison.csv")

if __name__ == "__main__":
    main()