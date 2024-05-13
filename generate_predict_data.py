import csv
import numpy as np

# Settings for prediction
num_sequences = 8  # Number of sequences to generate
start_ids = np.arange(1, 17, 8)  # Possible starting IDs that are multiples of 8
max_steps = 5  # Maximum number of steps in each sequence

# Seed for reproducibility
np.random.seed(42)

all_data = []

# Generate prediction data
for _ in range(num_sequences):
    start = np.random.choice(start_ids)
    total_steps = max_steps - 1 #np.random.randint(4, 7)  # Randomly choose between 4 and 6 steps inclusive
    sequence = [start]
    used_steps = set(sequence)

    while len(sequence) <= total_steps:
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

# Write prediction data to CSV file
predict_file_path = 'predict_data.csv'
with open(predict_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"step_{i+1}" for i in range(max_steps)])  # Header for max_steps columns
    for data in all_data:
        writer.writerow(data)

print(f"Prediction sample data generated and saved to {predict_file_path}")
