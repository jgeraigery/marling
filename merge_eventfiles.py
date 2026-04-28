# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_event_file(event_file):
    event_acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
    event_acc.Reload()
    
    tags = event_acc.Tags()["scalars"]
    data = {tag: [] for tag in tags}
    
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    
    return data

def merge_event_files(original_event_file, new_event_file, output_file):
    original_data = load_event_file(original_event_file)
    new_data = load_event_file(new_event_file)
    
    # Determine the maximum step and wall time in the original event file
    max_step = {}
    for tag in original_data.keys():
        # initialize the max_step and max_wall_time
        max_step[tag] = 0
        if original_data[tag]:
            max_step[tag] = original_data[tag][-1][0]
            
    # Offset the steps and wall times in the new event file
    for tag in new_data.keys():
        for i in range(len(new_data[tag])):
            step, value = new_data[tag][i]
            new_data[tag][i] = (step + max_step[tag], value)
            
    # Merge the data
    merged_data = {tag: [] for tag in original_data.keys()}
    for tag in original_data.keys():
        if tag not in new_data.keys():
            merged_data[tag] = original_data[tag]
        else:
            merged_data[tag] = original_data[tag] + new_data[tag]
            merged_data[tag].sort(key=lambda x: x[0])
            
    tf.compat.v1.disable_v2_behavior()
    metrics = [
    'Rollout/agent_0/reward',
    'Rollout/agent_0/win',
    'Rollout/agent_0/lose',
    'Training/agent_0/value_loss',
    'Training/agent_0/policy_loss',
    'Training/agent_0/entropy'
    ]

    with tf.compat.v1.summary.FileWriter(output_file) as writer:
        for tag in metrics:
            for step, value in merged_data[tag]:
                summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
                writer.add_summary(summary, global_step=step)
        writer.flush()
    
    writer.close()
    

if __name__ == "__main__":
    file1 = 'path/to/first/eventfile'
    file2 = 'path/to/second/eventfile'
    output_file = 'path/to/output/directory/'

    merge_event_files(file1, file2, output_file)
