#!/usr/bin/env python3
"""
Submit Block Discrete CQL training jobs
"""

import subprocess
import time
import os

# Configuration
bins_list = [3, 5, 10]
alphas_list = [0.0, 0.001, 0.01]
max_concurrent = 3

# Create all job combinations
jobs = [(b, a) for b in bins_list for a in alphas_list]

print(f"Submitting {len(jobs)} jobs (max {max_concurrent} concurrent)")
os.makedirs('logs/block_discrete', exist_ok=True)

# Submit jobs with concurrency limit
for i in range(0, len(jobs), max_concurrent):
    batch = jobs[i:i+max_concurrent]
    
    for bins, alpha in batch:
        log_file = f'logs/block_discrete/alpha{alpha:.4f}_bins{bins}.out'
        cmd = f'nohup python3 -u run_block_discrete_cql_allalphas.py --single_alpha {alpha} --vp2_bins {bins} > {log_file} 2>&1 &'
        subprocess.Popen(cmd, shell=True, cwd='/home/ubuntu/code/ucsf_rl')
        print(f"Started: bins={bins}, alpha={alpha}")
    
    # Wait for batch to complete if not last batch
    if i + max_concurrent < len(jobs):
        print(f"Waiting for batch to complete...")
        time.sleep(1800)  # Wait 30 minutes before next batch

print("\nAll jobs submitted. Check logs in logs/block_discrete/")