import random
import psutil
import os
import json
import csv

def generate_sequence(length, target_mean, tolerance):
    num_ones = int(length * target_mean)
    num_zeros = length - num_ones
    sequence = [1] * num_ones + [0] * num_zeros
    random.shuffle(sequence)
    return sequence

def within_tolerance(sequence, target_mean, tolerance):
    mean = sum(sequence) / len(sequence)
    return abs(mean - target_mean) <= tolerance

def save_sequences_txt(sequences, batch_number):
    filename = f'sequences_batch_{batch_number}.txt'
    with open(filename, 'w') as f:
        for seq in sequences:
            f.write(''.join(map(str, seq)) + '\n')

def save_sequences_json(sequences, batch_number):
    filename = f'sequences_batch_{batch_number}.json'
    with open(filename, 'w') as f:
        json.dump([list(seq) for seq in sequences], f)

def save_sequences_csv(sequences, batch_number):
    filename = f'sequences_batch_{batch_number}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sequences)

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def generate_and_save_sequences(target_mean=0.67, tolerance=0.8, max_memory_mb=500, target_sequences=1.6*(10**60)):
    sequence_length = 80
    batch_number = 0
    sequences = []
    valid_sequences = 0

    while valid_sequences < target_sequences:
        seq = generate_sequence(sequence_length, target_mean, tolerance)
        if within_tolerance(seq, target_mean, tolerance):
            sequences.append(seq)
            valid_sequences += 1
        
        if memory_usage() > max_memory_mb:
            #save_sequences_txt(sequences, batch_number)
            #save_sequences_json(sequences, batch_number)
            save_sequences_csv(sequences, batch_number)
            sequences = []  # Clear the list to free up memory
            batch_number += 1
            print(f'Batch: {batch_number}')
    
    # Save any remaining sequences
    if sequences:
        save_sequences_txt(sequences, batch_number)
        save_sequences_json(sequences, batch_number)
        save_sequences_csv(sequences, batch_number)

    

if __name__ == "__main__":
    generate_and_save_sequences()
