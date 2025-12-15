import csv
import os
from collections import Counter

CSV_FILE = 'master_dataset.csv'

def check_balance():
    print(f"Reading {CSV_FILE}...")
    
    label_counts = Counter()
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Replicate dataset.py logic exactly
            # Determine State Label
            v2 = row['v2_label']
            v1 = row['v1_label']
            
            state_str = 'Unlabeled'
            if v2 and v2 != 'Unlabeled':
                state_str = v2
            elif v1 and v1 != 'Unlabeled':
                state_str = v1
            
            label_counts[state_str] += 1
            
    print("\n--- Raw State Counts ---")
    for label, count in label_counts.most_common():
        percentage = (count / label_counts.total()) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    check_balance()
