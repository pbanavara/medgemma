"""
SRTR Trajectory Dataset for ODE-RNN Model.

Converts synthetic SRTR data into the format expected by the ClinicalODERNN model.
Supports multi-variate trajectories (4 features) for kidney and liver candidates.

Features:
- Kidney: Serum Creatinine, Albumin, GFR, Dialysis Status
- Liver: Bilirubin, INR, Serum Creatinine, MELD Score
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


# Feature configurations for each organ
KIDNEY_FEATURES = ['CANHX_SERUM_CREAT', 'CANHX_ALBUMIN', 'CANHX_GFR', 'CANHX_DIAL']
LIVER_FEATURES = ['CANHX_BILI', 'CANHX_INR', 'CANHX_SERUM_CREAT', 'CANHX_SRTR_LAB_MELD']

# Feature normalization parameters (mean, std) based on clinical ranges
# These help stabilize training by normalizing features to similar scales
FEATURE_NORMALIZATION = {
    # Kidney features
    'CANHX_SERUM_CREAT': (6.0, 3.0),      # Dialysis patients: mean ~6-8, std ~2-3
    'CANHX_ALBUMIN': (3.5, 0.6),           # Normal ~3.5-5, low <3.5
    'CANHX_GFR': (15.0, 8.0),              # ESRD patients: typically <15
    'CANHX_DIAL': (2.0, 1.0),              # Categorical: 1=No, 2=Hemo, 3=Peritoneal
    # Liver features
    'CANHX_BILI': (5.0, 6.0),              # Highly variable: 1-30+
    'CANHX_INR': (1.5, 0.5),               # Normal ~1, elevated 1.5-3+
    'CANHX_SRTR_LAB_MELD': (18.0, 8.0),   # Range 6-40
}


def sas_date_to_hours_since_listing(sas_date: int, listing_sas_date: int) -> float:
    """
    Convert SAS date to hours since listing.

    SAS dates are days since 1960-01-01.
    We convert the difference to hours.

    Args:
        sas_date: The status update SAS date
        listing_sas_date: The candidate's listing SAS date

    Returns:
        Hours since listing (float)
    """
    days_since_listing = sas_date - listing_sas_date
    return float(days_since_listing * 24)


class SRTRTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for SRTR transplant waitlist trajectories.

    Loads candidate and status history data, constructs multi-variate
    time series, and creates binary mortality labels.

    Args:
        cand_df: Candidate DataFrame (CAND_KIPA or CAND_LIIN with metadata)
        stathist_df: Status history DataFrame (STATHIST_KIPA or STATHIST_LIIN)
        organ: 'kidney' or 'liver'
        max_seq_len: Maximum sequence length (default 50)
        mortality_horizon_days: Days for mortality label (default 90)
        normalize_features: Whether to normalize features (default True)
    """

    def __init__(
        self,
        cand_df: pd.DataFrame,
        stathist_df: pd.DataFrame,
        organ: str = 'kidney',
        max_seq_len: int = 50,
        mortality_horizon_days: int = 90,
        normalize_features: bool = True
    ):
        self.organ = organ.lower()
        self.max_seq_len = max_seq_len
        self.mortality_horizon_days = mortality_horizon_days
        self.normalize_features = normalize_features

        # Select features based on organ
        if self.organ == 'kidney':
            self.features = KIDNEY_FEATURES
        elif self.organ == 'liver':
            self.features = LIVER_FEATURES
        else:
            raise ValueError(f"Unknown organ: {organ}. Must be 'kidney' or 'liver'.")

        self.input_dim = len(self.features)

        # Process data
        self.samples = self._prepare_samples(cand_df, stathist_df)

        print(f"Created {self.organ} dataset with {len(self.samples)} samples")
        print(f"  Features ({self.input_dim}): {self.features}")
        print(f"  Max sequence length: {self.max_seq_len}")
        print(f"  Mortality horizon: {self.mortality_horizon_days} days")

        # Print class distribution
        labels = [s['label'] for s in self.samples]
        n_positive = sum(labels)
        print(f"  Class distribution: {n_positive} deaths ({n_positive/len(labels)*100:.1f}%), "
              f"{len(labels)-n_positive} non-deaths ({(len(labels)-n_positive)/len(labels)*100:.1f}%)")

    def _prepare_samples(
        self,
        cand_df: pd.DataFrame,
        stathist_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Prepare samples by joining candidate and status history data.

        Returns list of dicts with:
        - values: (seq_len, input_dim) tensor
        - times: (seq_len,) tensor in hours since listing
        - mask: (seq_len,) tensor
        - label: 0 or 1
        - px_id: patient ID
        """
        samples = []

        # Group status history by patient
        grouped = stathist_df.groupby('PX_ID')

        for px_id, group in grouped:
            # Get candidate info
            cand_row = cand_df[cand_df['PX_ID'] == px_id]
            if len(cand_row) == 0:
                continue
            cand_row = cand_row.iloc[0]

            # Get listing date
            listing_sas = cand_row['CAN_LISTING_DT']

            # Sort status history by date
            group = group.sort_values('CANHX_BEGIN_DT')

            # Extract time points (hours since listing)
            times = []
            for _, row in group.iterrows():
                hours = sas_date_to_hours_since_listing(row['CANHX_BEGIN_DT'], listing_sas)
                times.append(hours)

            # Extract feature values
            values = []
            for _, row in group.iterrows():
                feat_values = []
                for feat in self.features:
                    val = row.get(feat, 0.0)
                    if pd.isna(val):
                        val = 0.0

                    # Normalize if enabled
                    if self.normalize_features and feat in FEATURE_NORMALIZATION:
                        mean, std = FEATURE_NORMALIZATION[feat]
                        val = (val - mean) / std

                    feat_values.append(float(val))
                values.append(feat_values)

            # Create label: death within mortality_horizon_days
            outcome = cand_row.get('_outcome', 'WAITING')
            days_on_waitlist = cand_row.get('_days_on_waitlist', 9999)

            if outcome == 'DEATH' and days_on_waitlist <= self.mortality_horizon_days:
                label = 1
            else:
                label = 0

            # Truncate or pad sequence
            seq_len = len(times)

            if seq_len > self.max_seq_len:
                # Keep most recent observations
                times = times[-self.max_seq_len:]
                values = values[-self.max_seq_len:]
                mask = [1] * self.max_seq_len
            else:
                # Pad with zeros
                pad_len = self.max_seq_len - seq_len
                times = times + [0.0] * pad_len
                values = values + [[0.0] * self.input_dim] * pad_len
                mask = [1] * seq_len + [0] * pad_len

            samples.append({
                'values': values,
                'times': times,
                'mask': mask,
                'label': label,
                'px_id': px_id
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        return {
            'values': torch.tensor(sample['values'], dtype=torch.float32),
            'times': torch.tensor(sample['times'], dtype=torch.float32),
            'mask': torch.tensor(sample['mask'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.float32),
        }


def load_srtr_datasets(
    data_dir: Union[str, Path],
    organ: str = 'kidney',
    max_seq_len: int = 50,
    mortality_horizon_days: int = 90,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[SRTRTrajectoryDataset, SRTRTrajectoryDataset, SRTRTrajectoryDataset]:
    """
    Load SRTR data and create train/val/test datasets.

    Args:
        data_dir: Path to directory containing SRTR CSV files
        organ: 'kidney' or 'liver'
        max_seq_len: Maximum trajectory length
        mortality_horizon_days: Days for mortality label
        train_split, val_split, test_split: Dataset split ratios
        seed: Random seed for reproducibility
        max_samples: Maximum number of patients to use (None = all)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)

    # Load appropriate files based on organ
    if organ.lower() == 'kidney':
        cand_file = data_dir / 'CAND_KIPA.csv'
        stathist_file = data_dir / 'STATHIST_KIPA.csv'
    elif organ.lower() == 'liver':
        cand_file = data_dir / 'CAND_LIIN.csv'
        stathist_file = data_dir / 'STATHIST_LIIN.csv'
    else:
        raise ValueError(f"Unknown organ: {organ}")

    print(f"Loading {organ} data from {data_dir}...")

    # Load DataFrames
    cand_df = pd.read_csv(cand_file)
    stathist_df = pd.read_csv(stathist_file)

    print(f"  Loaded {len(cand_df)} candidates, {len(stathist_df)} status records")

    # We need the internal metadata columns for labels
    # Check if _outcome exists, if not, derive from CAN_REM_CD
    if '_outcome' not in cand_df.columns:
        # Derive outcome from removal code
        def derive_outcome(row):
            rem_cd = row.get('CAN_REM_CD')
            if pd.isna(rem_cd):
                return 'WAITING'
            elif rem_cd == 4:
                return 'DEATH'
            elif rem_cd in [8, 9, 14]:
                return 'TRANSPLANT'
            else:
                return 'REMOVED'

        cand_df['_outcome'] = cand_df.apply(derive_outcome, axis=1)

        # Derive days on waitlist
        def derive_days(row):
            if pd.isna(row.get('CAN_REM_DT')) or pd.isna(row.get('CAN_LISTING_DT')):
                return 9999
            return row['CAN_REM_DT'] - row['CAN_LISTING_DT']

        cand_df['_days_on_waitlist'] = cand_df.apply(derive_days, axis=1)

    # Split candidates into train/val/test
    np.random.seed(seed)
    px_ids = cand_df['PX_ID'].unique()
    np.random.shuffle(px_ids)

    # Limit samples if requested (0 or None means use all)
    if max_samples is not None and max_samples > 0 and len(px_ids) > max_samples:
        px_ids = px_ids[:max_samples]
        print(f"  Subsampled to {max_samples} patients")

    n_total = len(px_ids)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_ids = set(px_ids[:n_train])
    val_ids = set(px_ids[n_train:n_train + n_val])
    test_ids = set(px_ids[n_train + n_val:])

    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Create subset DataFrames
    train_cand = cand_df[cand_df['PX_ID'].isin(train_ids)]
    train_stathist = stathist_df[stathist_df['PX_ID'].isin(train_ids)]

    val_cand = cand_df[cand_df['PX_ID'].isin(val_ids)]
    val_stathist = stathist_df[stathist_df['PX_ID'].isin(val_ids)]

    test_cand = cand_df[cand_df['PX_ID'].isin(test_ids)]
    test_stathist = stathist_df[stathist_df['PX_ID'].isin(test_ids)]

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = SRTRTrajectoryDataset(
        train_cand, train_stathist, organ, max_seq_len, mortality_horizon_days
    )

    print("\nCreating validation dataset...")
    val_dataset = SRTRTrajectoryDataset(
        val_cand, val_stathist, organ, max_seq_len, mortality_horizon_days
    )

    print("\nCreating test dataset...")
    test_dataset = SRTRTrajectoryDataset(
        test_cand, test_stathist, organ, max_seq_len, mortality_horizon_days
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_dir: Union[str, Path],
    organ: str = 'kidney',
    batch_size: int = 32,
    max_seq_len: int = 50,
    mortality_horizon_days: int = 90,
    num_workers: int = 16,
    max_samples: Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to get DataLoaders for training.

    Args:
        max_samples: Maximum number of patients to use (None = all)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = load_srtr_datasets(
        data_dir, organ, max_seq_len, mortality_horizon_days,
        max_samples=max_samples, **kwargs
    )

    pin_memory = num_workers > 0  # Enable pin_memory for CUDA with multiprocessing

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Default to synthetic data directory
    data_dir = Path(__file__).parent.parent.parent / "synthetic-srtr-data" / "srtr_synthetic"

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    print(f"Loading data from: {data_dir}")
    print("=" * 60)

    # Test kidney dataset
    print("\n--- KIDNEY DATASET ---")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir,
            organ='kidney',
            batch_size=4,
            max_seq_len=50,
            mortality_horizon_days=90
        )

        # Get a sample batch
        batch = next(iter(train_loader))
        print(f"\nSample batch shapes:")
        print(f"  values: {batch['values'].shape}")  # (batch, seq, features)
        print(f"  times: {batch['times'].shape}")    # (batch, seq)
        print(f"  mask: {batch['mask'].shape}")      # (batch, seq)
        print(f"  label: {batch['label'].shape}")    # (batch,)

        print(f"\nSample time points (hours): {batch['times'][0, :10].tolist()}")
        print(f"Sample values (first 3 time points):\n{batch['values'][0, :3, :]}")
        print(f"Sample label: {batch['label'][0].item()}")

    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Run the SRTR generator first to create synthetic data.")

    # Test liver dataset
    print("\n--- LIVER DATASET ---")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir,
            organ='liver',
            batch_size=4,
            max_seq_len=50,
            mortality_horizon_days=90
        )

        batch = next(iter(train_loader))
        print(f"\nSample batch shapes:")
        print(f"  values: {batch['values'].shape}")
        print(f"  times: {batch['times'].shape}")
        print(f"  label: {batch['label'].shape}")

    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
