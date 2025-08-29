"""
Train/Validation/Test split utility for patient-level splitting
"""

import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

import data_config as config
from data_loader_simple import DataLoader

# Set global random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

class DataSplitter:
    """
    Handles train/validation/test splitting at the patient level
    IMPORTANT: We split by patient, not by timestep, to avoid data leakage
    """
    
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Initialize data splitter
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.train_patients = None
        self.val_patients = None
        self.test_patients = None
        
        # Set seeds for this instance
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
    def split_patients(self, patient_ids: np.ndarray, 
                      train_ratio: float = config.TRAIN_RATIO,
                      val_ratio: float = config.VAL_RATIO,
                      test_ratio: float = config.TEST_RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split patient IDs into train/val/test sets
        
        Args:
            patient_ids: Array of all patient IDs
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)
            
        Returns:
            Tuple of (train_patients, val_patients, test_patients)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            patient_ids,
            test_size=test_ratio,
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        # Calculate validation size relative to train+val
        val_size_relative = val_ratio / (train_ratio + val_ratio)
        
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_size_relative,
            random_state=self.random_seed
        )
        
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients
        
        return train_patients, val_patients, test_patients
    
    def get_split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the full dataset based on patient splits
        
        Args:
            data: Full DataFrame with all patients
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.train_patients is None:
            raise ValueError("Must call split_patients() first")
        
        train_data = data[data[config.PATIENT_ID_COL].isin(self.train_patients)]
        val_data = data[data[config.PATIENT_ID_COL].isin(self.val_patients)]
        test_data = data[data[config.PATIENT_ID_COL].isin(self.test_patients)]
        
        return train_data, val_data, test_data
    
    def get_split_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Get statistics about the splits
        
        Args:
            data: Full DataFrame
            
        Returns:
            Dictionary with split statistics
        """
        if self.train_patients is None:
            raise ValueError("Must call split_patients() first")
        
        train_data, val_data, test_data = self.get_split_data(data)
        
        stats = {
            'train': {
                'n_patients': len(self.train_patients),
                'n_timesteps': len(train_data),
                'n_died': sum(train_data.groupby(config.PATIENT_ID_COL)[config.DEATH_COL].last() == 1),
                'mortality_rate': 0
            },
            'val': {
                'n_patients': len(self.val_patients),
                'n_timesteps': len(val_data),
                'n_died': sum(val_data.groupby(config.PATIENT_ID_COL)[config.DEATH_COL].last() == 1),
                'mortality_rate': 0
            },
            'test': {
                'n_patients': len(self.test_patients),
                'n_timesteps': len(test_data),
                'n_died': sum(test_data.groupby(config.PATIENT_ID_COL)[config.DEATH_COL].last() == 1),
                'mortality_rate': 0
            }
        }
        
        # Calculate mortality rates
        for split in ['train', 'val', 'test']:
            stats[split]['mortality_rate'] = stats[split]['n_died'] / stats[split]['n_patients']
        
        return stats


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("="*70)
    print(" DATA SPLITTING TEST")
    print("="*70)
    
    # Load data
    loader = DataLoader(verbose=False)
    data = loader.load_data()
    patient_ids = loader.get_patient_ids()
    
    print(f"\nTotal patients: {len(patient_ids)}")
    print(f"Total timesteps: {len(data)}")
    
    # Create splitter and split patients
    splitter = DataSplitter(random_seed=42)
    train_patients, val_patients, test_patients = splitter.split_patients(patient_ids)
    
    print(f"\nPatient splits:")
    print(f"  Train: {len(train_patients)} ({len(train_patients)/len(patient_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_patients)} ({len(val_patients)/len(patient_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_patients)} ({len(test_patients)/len(patient_ids)*100:.1f}%)")
    
    # Get split data
    train_data, val_data, test_data = splitter.get_split_data(data)
    
    print(f"\nTimestep splits:")
    print(f"  Train: {len(train_data)} timesteps")
    print(f"  Val:   {len(val_data)} timesteps")
    print(f"  Test:  {len(test_data)} timesteps")
    
    # Get detailed statistics
    stats = splitter.get_split_statistics(data)
    
    print("\n" + "="*70)
    print(" DETAILED SPLIT STATISTICS")
    print("="*70)
    
    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        print(f"\n{split_name.upper()} SET:")
        print(f"  Patients:       {s['n_patients']}")
        print(f"  Timesteps:      {s['n_timesteps']}")
        print(f"  Avg traj len:   {s['n_timesteps']/s['n_patients']:.1f}")
        print(f"  Died:           {s['n_died']}")
        print(f"  Mortality rate: {s['mortality_rate']*100:.1f}%")
    
    # Verify no patient overlap
    print("\n" + "="*70)
    print(" VERIFICATION: NO PATIENT OVERLAP")
    print("="*70)
    
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    
    print(f"Train ∩ Val:  {len(train_set & val_set)} patients (should be 0)")
    print(f"Train ∩ Test: {len(train_set & test_set)} patients (should be 0)")
    print(f"Val ∩ Test:   {len(val_set & test_set)} patients (should be 0)")
    
    if len(train_set & val_set) == 0 and len(train_set & test_set) == 0 and len(val_set & test_set) == 0:
        print("\n✅ No patient overlap - splits are valid!")
    else:
        print("\n❌ Patient overlap detected - this is a bug!")
    
    # Show sample patients from each split
    print("\n" + "="*70)
    print(" SAMPLE PATIENT IDS")
    print("="*70)
    
    print(f"First 5 train patients: {train_patients[:5]}")
    print(f"First 5 val patients:   {val_patients[:5]}")
    print(f"First 5 test patients:  {test_patients[:5]}")