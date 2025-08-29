#!/usr/bin/env python3
"""
Debug script to investigate VP2 (norepinephrine) values in the data
"""

import numpy as np
import pandas as pd
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

print("="*70)
print(" INVESTIGATING VP2 (NOREPINEPHRINE) VALUES")
print("="*70)

# First, check raw data
print("\n1. CHECKING RAW DATA")
print("-"*70)
df = pd.read_csv('sample_data_oviss.csv')

# Check if norepinephrine column exists
if 'norepinephrine' in df.columns:
    norep = df['norepinephrine']
    print(f"Norepinephrine column found")
    print(f"  Total rows: {len(norep)}")
    print(f"  Non-null values: {norep.notna().sum()}")
    print(f"  Null values: {norep.isna().sum()}")
    print(f"\nStatistics for non-null values:")
    print(f"  Min: {norep.min():.4f}")
    print(f"  Max: {norep.max():.4f}")
    print(f"  Mean: {norep.mean():.4f}")
    print(f"  Median: {norep.median():.4f}")
    print(f"  Std: {norep.std():.4f}")
    
    # Check value distribution
    print(f"\nValue distribution:")
    print(f"  Values == 0: {(norep == 0).sum()}")
    print(f"  Values in (0, 0.5]: {((norep > 0) & (norep <= 0.5)).sum()}")
    print(f"  Values in (0.5, 1]: {((norep > 0.5) & (norep <= 1)).sum()}")
    print(f"  Values in (1, 10]: {((norep > 1) & (norep <= 10)).sum()}")
    print(f"  Values > 10: {(norep > 10).sum()}")
    
    # Sample some high values
    high_values = norep[norep > 0.5].dropna()
    if len(high_values) > 0:
        print(f"\nSample of values > 0.5:")
        sample_high = high_values.sample(min(20, len(high_values)), random_state=42).sort_values(ascending=False)
        for val in sample_high:
            print(f"  {val:.4f}")
else:
    print("Norepinephrine column NOT found in raw data!")
    print("Available columns:", df.columns.tolist())

# Now check through pipeline
print("\n" + "="*70)
print("2. CHECKING THROUGH INTEGRATEDDATAPIPELINEV2")
print("-"*70)

pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
train_data, val_data, test_data = pipeline.prepare_data()

print("\nChecking VP2 values in pipeline output (dual model):")
print("Train data actions shape:", train_data['actions'].shape)

# VP2 is the second column in dual model actions
vp2_train = train_data['actions'][:, 1]
print(f"\nTrain VP2 statistics:")
print(f"  Min: {vp2_train.min():.4f}")
print(f"  Max: {vp2_train.max():.4f}")
print(f"  Mean: {vp2_train.mean():.4f}")
print(f"  Median: {np.median(vp2_train):.4f}")
print(f"  Std: {vp2_train.std():.4f}")

print(f"\nTrain VP2 distribution:")
print(f"  Values in [0, 0.5]: {((vp2_train >= 0) & (vp2_train <= 0.5)).sum()} ({((vp2_train >= 0) & (vp2_train <= 0.5)).mean()*100:.1f}%)")
print(f"  Values in (0.5, 1]: {((vp2_train > 0.5) & (vp2_train <= 1)).sum()} ({((vp2_train > 0.5) & (vp2_train <= 1)).mean()*100:.1f}%)")
print(f"  Values > 1: {(vp2_train > 1).sum()} ({(vp2_train > 1).mean()*100:.1f}%)")

# Sample some random batches to see what we get
print("\n" + "="*70)
print("3. CHECKING RANDOM BATCHES")
print("-"*70)

for i in range(3):
    batch = pipeline.get_batch(batch_size=32, split='train')
    vp2_batch = batch['actions'][:, 1]
    print(f"\nBatch {i+1}:")
    print(f"  VP2 range: [{vp2_batch.min():.4f}, {vp2_batch.max():.4f}]")
    print(f"  Values > 0.5: {(vp2_batch > 0.5).sum()} out of 32")
    if (vp2_batch > 0.5).any():
        high_vals = vp2_batch[vp2_batch > 0.5]
        print(f"  High values: {high_vals[:5]}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nThe norepinephrine values in the raw data appear to be in a different scale!")
print("They likely need to be normalized to mcg/kg/min in the [0, 0.5] range.")
print("Current data has values up to", norep.max() if 'norepinephrine' in df.columns else "unknown")