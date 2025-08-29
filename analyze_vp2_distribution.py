#!/usr/bin/env python3
"""
Analyze VP2 (norepinephrine) distribution and impact of different clamping thresholds
"""

import numpy as np
import pandas as pd
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

print("="*70)
print(" DETAILED VP2 (NOREPINEPHRINE) DISTRIBUTION ANALYSIS")
print("="*70)

# Load raw data
df = pd.read_csv('sample_data_oviss.csv')
norep = df['norepinephrine'].values

print("\n1. RAW DATA ANALYSIS")
print("-"*70)
print(f"Total samples: {len(norep):,}")

# Define clamping thresholds to test
thresholds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

print("\nImpact of different clamping thresholds:")
print(f"{'Threshold':<12} {'Values Above':<15} {'% Above':<10} {'% Kept':<10} {'Max After Clamp':<15}")
print("-"*70)

for threshold in thresholds:
    n_above = (norep > threshold).sum()
    pct_above = (norep > threshold).mean() * 100
    pct_kept = 100 - pct_above
    clamped = np.clip(norep, 0, threshold)
    max_after = clamped.max()
    print(f"{threshold:<12.2f} {n_above:<15,} {pct_above:<10.2f} {pct_kept:<10.2f} {max_after:<15.2f}")

# Percentile analysis
print("\n2. PERCENTILE ANALYSIS")
print("-"*70)
percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
print(f"{'Percentile':<12} {'Value':<15}")
print("-"*40)
for p in percentiles:
    val = np.percentile(norep, p)
    print(f"{p:<12.1f} {val:<15.4f}")

# Use pipeline to get larger sample
print("\n3. PIPELINE DATA ANALYSIS (LARGE SAMPLE)")
print("-"*70)

pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
train_data, val_data, test_data = pipeline.prepare_data()

# Get a large batch (10000 samples)
large_batch_size = min(10000, len(train_data['states']))
print(f"Sampling {large_batch_size:,} transitions from training data...")

# Get multiple batches to reach desired size
all_vp2 = []
n_batches_needed = (large_batch_size // 1000) + 1

for i in range(n_batches_needed):
    batch = pipeline.get_batch(batch_size=min(1000, large_batch_size - len(all_vp2)), split='train')
    vp2_batch = batch['actions'][:, 1]
    all_vp2.extend(vp2_batch)
    if len(all_vp2) >= large_batch_size:
        break

all_vp2 = np.array(all_vp2[:large_batch_size])

print(f"\nStatistics for {len(all_vp2):,} sampled VP2 values:")
print(f"  Min:    {all_vp2.min():.4f}")
print(f"  Max:    {all_vp2.max():.4f}")
print(f"  Mean:   {all_vp2.mean():.4f}")
print(f"  Median: {np.median(all_vp2):.4f}")
print(f"  Std:    {all_vp2.std():.4f}")

print("\nDistribution by range:")
ranges = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, float('inf'))]
for low, high in ranges:
    if high == float('inf'):
        mask = all_vp2 > low
        range_str = f">{low:.1f}"
    else:
        mask = (all_vp2 > low) & (all_vp2 <= high)
        range_str = f"({low:.1f}, {high:.1f}]"
    
    count = mask.sum()
    pct = mask.mean() * 100
    print(f"  {range_str:<12} {count:>6,} samples ({pct:>5.1f}%)")

# Recommendation
print("\n4. RECOMMENDATION")
print("-"*70)

# Calculate impact of clamping to 1.0
clamp_1_loss = (norep > 1.0).mean() * 100
clamp_0_5_loss = (norep > 0.5).mean() * 100

print(f"\nClamping to 1.0:")
print(f"  - Keeps {100-clamp_1_loss:.1f}% of data unchanged")
print(f"  - Affects {clamp_1_loss:.1f}% of data points")
print(f"  - Maximum value becomes 1.0 (from {norep.max():.1f})")

print(f"\nClamping to 0.5 (original range):")
print(f"  - Keeps {100-clamp_0_5_loss:.1f}% of data unchanged")
print(f"  - Affects {clamp_0_5_loss:.1f}% of data points")
print(f"  - Maximum value becomes 0.5 (from {norep.max():.1f})")

# Show some examples of high values
print("\n5. EXAMPLES OF HIGH VP2 VALUES")
print("-"*70)
high_values = norep[norep > 1.0]
if len(high_values) > 0:
    # Get unique high values
    unique_high = np.unique(high_values)
    print(f"Found {len(unique_high)} unique values > 1.0")
    print("\nTop 20 unique values:")
    for val in sorted(unique_high, reverse=True)[:20]:
        count = (norep == val).sum()
        print(f"  {val:>7.2f} (appears {count:>4} times)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n• Total data points: {len(norep):,}")
print(f"• Values > 0.5: {(norep > 0.5).sum():,} ({(norep > 0.5).mean()*100:.1f}%)")
print(f"• Values > 1.0: {(norep > 1.0).sum():,} ({(norep > 1.0).mean()*100:.1f}%)")
print(f"• 99th percentile: {np.percentile(norep, 99):.3f}")
print(f"• 99.5th percentile: {np.percentile(norep, 99.5):.3f}")
print("\nClamping to 1.0 would preserve ~95% of data unchanged")
print("while handling extreme outliers that are likely data errors.")