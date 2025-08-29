"""
Comprehensive script to run all discrete CQL experiments with different alpha and bin combinations
Runs in background with proper logging
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path


def setup_logging_directory():
    """Create logging directory structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f'logs/discrete_cql_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, timestamp


def create_single_experiment_script(alpha, vp2_bins, experiment_dir, log_dir):
    """Create a Python script for a single experiment"""
    script_content = f"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_unified_dual_cql_block_discrete import (
    DualDiscreteCQL, 
    prepare_data, 
    train_single_cql,
    ContinuousQNetwork
)

def main():
    print("="*80)
    print(f"Starting experiment: alpha={alpha}, VP2_bins={vp2_bins}")
    print("="*80)
    
    # Load and prepare data
    data = prepare_data('sample_data_oviss.csv', {vp2_bins})
    
    # Create experiment directory
    exp_dir = os.path.join('{experiment_dir}', 'alpha{alpha}_bins{vp2_bins}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Train model with 100 epochs
    history = train_single_cql(
        data=data,
        alpha={alpha},
        vp2_bins={vp2_bins},
        epochs=100,  # Full training
        batch_size=128,
        save_dir=exp_dir
    )
    
    # Save results
    result = {{
        'alpha': {alpha},
        'vp2_bins': {vp2_bins},
        'best_val_loss': history['best_val_loss'],
        'final_train_q_loss': history['train_q_loss'][-1] if history['train_q_loss'] else None,
        'final_train_cql_loss': history['train_cql_loss'][-1] if history['train_cql_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'completed': datetime.now().isoformat()
    }}
    
    result_file = os.path.join(exp_dir, 'experiment_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\\n✅ Experiment completed: alpha={alpha}, bins={vp2_bins}")
    print(f"   Best val loss: {{history['best_val_loss']:.4f}}")
    
    return result

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR in experiment alpha={alpha}, bins={vp2_bins}: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
    
    # Save script
    script_name = f"experiment_alpha{alpha}_bins{vp2_bins}.py"
    script_path = log_dir / script_name
    script_path.write_text(script_content)
    
    return script_path


def launch_experiment(script_path, alpha, vp2_bins, log_dir):
    """Launch a single experiment as a subprocess"""
    log_file = log_dir / f"alpha{alpha}_bins{vp2_bins}.log"
    err_file = log_dir / f"alpha{alpha}_bins{vp2_bins}.err"
    
    cmd = [
        sys.executable, '-u',  # Unbuffered output
        str(script_path)
    ]
    
    with open(log_file, 'w') as stdout, open(err_file, 'w') as stderr:
        process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            cwd='/home/ubuntu/code/ucsf_rl'
        )
    
    return process, log_file, err_file


def main():
    print("="*80)
    print("DISCRETE CQL COMPREHENSIVE EXPERIMENT RUNNER")
    print("="*80)
    
    # Configuration
    alpha_values = [0.0, 0.001, 0.01]
    vp2_bins_options = [5, 10, 30, 50]
    
    # Setup logging
    log_dir, timestamp = setup_logging_directory()
    experiment_dir = f'experiment/discrete_cql_{timestamp}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Log directory: {log_dir}")
    print(f"\nRunning {len(alpha_values) * len(vp2_bins_options)} experiments:")
    print(f"  Alpha values: {alpha_values}")
    print(f"  VP2 bins: {vp2_bins_options}")
    
    # Create master config file
    config = {
        'timestamp': timestamp,
        'experiment_dir': experiment_dir,
        'log_dir': str(log_dir),
        'alpha_values': alpha_values,
        'vp2_bins_options': vp2_bins_options,
        'total_experiments': len(alpha_values) * len(vp2_bins_options),
        'experiments': []
    }
    
    # Launch all experiments
    processes = []
    experiment_id = 0
    
    for vp2_bins in vp2_bins_options:
        for alpha in alpha_values:
            experiment_id += 1
            print(f"\n[{experiment_id}/{config['total_experiments']}] Launching alpha={alpha}, bins={vp2_bins}")
            
            # Create script for this experiment
            script_path = create_single_experiment_script(
                alpha, vp2_bins, experiment_dir, log_dir
            )
            
            # Launch experiment
            process, log_file, err_file = launch_experiment(
                script_path, alpha, vp2_bins, log_dir
            )
            
            exp_info = {
                'id': experiment_id,
                'alpha': alpha,
                'vp2_bins': vp2_bins,
                'pid': process.pid,
                'log_file': str(log_file),
                'err_file': str(err_file),
                'script': str(script_path),
                'status': 'running',
                'started': datetime.now().isoformat()
            }
            
            config['experiments'].append(exp_info)
            processes.append((process, exp_info))
            
            print(f"   Started with PID: {process.pid}")
            print(f"   Logs: {log_file}")
            
            # Small delay between launches to avoid overwhelming the system
            time.sleep(2)
    
    # Save master config
    config_file = log_dir / 'master_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS LAUNCHED")
    print("="*80)
    print(f"\nMaster config: {config_file}")
    print(f"Monitor progress with: python3 monitor_discrete_cql.py {log_dir}")
    print("\nExperiments running in background. You can safely close this terminal.")
    
    # Create PID file for monitoring
    pid_file = log_dir / 'experiment_pids.txt'
    with open(pid_file, 'w') as f:
        for process, exp_info in processes:
            f.write(f"{process.pid} alpha{exp_info['alpha']}_bins{exp_info['vp2_bins']}\n")
    
    print(f"\nPID file: {pid_file}")
    
    return log_dir


if __name__ == "__main__":
    log_dir = main()
    print(f"\n✅ All experiments launched successfully!")
    print(f"📁 Results will be saved in: {log_dir}")