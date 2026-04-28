# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import os
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer import event_file_writer
import time

def add_label_to_event_file(input_event_file, output_event_file, label, value=1.0):
    """
    Adds a new label/tag to an existing TensorBoard event file
    
    Args:
        input_event_file: Path to the original event file
        output_event_file: Path to write the augmented event file
        label: The label/tag to add (e.g., "Training/agent_0/...")
        value: Value to assign to the label (default: 1.0)
    """
    print(f"Processing {input_event_file} -> {output_event_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_event_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a writer for the output file
    writer = event_file_writer.EventFileWriter(output_event_file)
    
    # Process events from the input file
    loader = event_file_loader.EventFileLoader(input_event_file)
    
    # Keep track of wall time for the new event
    last_wall_time = time.time()
    
    # Find the last step in the file to add our new event after it
    max_step = 0
    for event in loader.Load():
        # Copy original event to output file
        writer.add_event(event)
        
        # Update last wall time and max step
        if event.HasField('summary'):
            last_wall_time = event.wall_time
            for val in event.summary.value:
                if hasattr(val, 'step') and val.step > max_step:
                    max_step = val.step
    
    # Create a new event with the additional label
    new_event = event_pb2.Event(
        step=max_step,
        wall_time=last_wall_time
    )
    
    from tensorboard.compat.proto import summary_pb2  # add this import

    # Then later, create the summary like so:
    summary = summary_pb2.Summary()
    summary.value.add(tag=label, simple_value=float(value))
    new_event.summary.CopyFrom(summary)
    
    # Write the new event
    writer.add_event(new_event)
    
    # Close the writer
    writer.close()
    print(f"Added label '{label}' with value {value} at step {max_step}")

def main():
    # Define directory and label pairs directly in the code
    # Format: (directory_pattern, label, value)
    # The script will find all event files matching the pattern
    dir_label_pairs = [
        # Example:
        ("./vanilla/2025-02-12-18-41-27/", "Training/agent_0/agent_0-default-13/V_mean", 1.0),
        ("./vanilla/2025-02-24-08-23-40/", "Training/agent_0/agent_0-default-13/V_mean", 1.0),
        ("./rnd/2025-02-20-14-33-59", "Training/agent_0/agent_0-default-13/V_mean", 1.0),
    ]
    
    # Process each directory pattern and label pair
    for dir_pattern, label, value in dir_label_pairs:
        # Find all matching event files
        event_files = []
        for root, dirs, files in os.walk(dir_pattern):
            for file in files:
                if file.startswith("event"):
                    event_files.append(os.path.join(root, file))
        
        # Process each event file
        for file_path in event_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue
            
            # Create output filename with _aug suffix
            base_dir = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name_parts = os.path.splitext(filename)
            output_file = os.path.join(base_dir, f"{name_parts[0]}_aug{name_parts[1]}")
            
            # Process the file
            add_label_to_event_file(file_path, output_file, label, value)

if __name__ == "__main__":
    main()