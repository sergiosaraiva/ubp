import csv
import numpy as np

# Settings
start_ids = np.arange(1, 17, 8)  # Starting IDs are multiples of 8
samples_per_start = 512  # 32 samples for each starting ID
max_steps = 7  # Maximum number of steps in each sequence

# Seed for reproducibility
np.random.seed(42)

all_data = []

# Generate data
for start in start_ids:
    for _ in range(samples_per_start):
        total_steps = max_steps  #np.random.randint(5, 8)  # Randomly choose between 5 and 7 steps inclusive
        sequence = [start]  # Start sequence with the initial multiple of 8
        used_steps = set(sequence)  # Track used steps to avoid duplicates

        while len(sequence) < total_steps:
            last_step = sequence[-1]
            if np.random.rand() < 0.5 and (last_step + 1) < start + 8:
                next_step = last_step + 1
            else:
                next_step = np.random.randint(start, start + 8)
            
            if next_step not in used_steps and next_step < start + 8:
                sequence.append(next_step)
                used_steps.add(next_step)
            else:
                possible_steps = list(set(range(start, start + 8)) - used_steps)
                if possible_steps:
                    chosen_step = np.random.choice(possible_steps)
                    sequence.append(chosen_step)
                    used_steps.add(chosen_step)

        # Pad the sequence to ensure it has max_steps elements
        sequence.extend([0] * (max_steps - len(sequence)))
        all_data.append(sequence)

# Write data to CSV file
file_path = 'training_data.csv'
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"step_{i+1}" for i in range(max_steps)])  # Header for max_steps columns
    for data in all_data:
        writer.writerow(data)

print(f"Sample data generated and saved to {file_path}")
