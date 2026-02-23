import torch
from torch.utils.data import Dataset, DataLoader
import re

class MIMICtrajectoryDataset(Dataset):
    def __init__(self, dataframe, max_len=20):
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def parse_trajectory(self, traj_string):
        """
        Converts string "24 (0h) -> 20 (4h)" into lists [24, 20] and [0, 4]
        """
        # Regex to find numbers. 
        # Pattern: look for a number, then space, then '(', then number, then 'h)'
        # But simpler is: Split by '->', then regex each chunk
        
        values = []
        times = []
        
        # Split into events
        events = traj_string.split(' -> ')
        
        for event in events:
            # Extract numbers using regex
            # Matches "24" and "0" from "24 (0h)"
            matches = re.findall(r"([\d\.]+)", event)
            if len(matches) >= 2:
                val = float(matches[0])
                t = float(matches[1])
                values.append(val)
                times.append(t)
                
        return values, times

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        traj_str = row['clinical_trajectory']
        label = row['label_escalation']
        
        # 1. Parse the string back into numbers
        vals, ts = self.parse_trajectory(traj_str)
        
        # 2. Truncate or Pad to fixed length (for batching)
        seq_len = len(vals)
        if seq_len > self.max_len:
            # Truncate (keep last 20 events as they are most recent)
            vals = vals[-self.max_len:]
            ts = ts[-self.max_len:]
            mask = [1] * self.max_len
        else:
            # Pad with zeros
            pad_len = self.max_len - seq_len
            vals = vals + [0] * pad_len
            ts = ts + [0] * pad_len
            mask = [1] * seq_len + [0] * pad_len # Mask tells model which are real data

        # 3. Convert to Tensors
        return {
            'values': torch.tensor(vals, dtype=torch.float32),
            'times': torch.tensor(ts, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32), # Attention Mask
            'label': torch.tensor(label, dtype=torch.float32)
        }

# Usage:
# dataset = MIMICtrajectoryDataset(training_data)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)