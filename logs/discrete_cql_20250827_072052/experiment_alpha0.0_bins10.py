
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
    print(f"Starting experiment: alpha=0.0, VP2_bins=10")
    print("="*80)
    
    # Load and prepare data
    data = prepare_data('sample_data_oviss.csv', 10)
    
    # Create experiment directory
    exp_dir = os.path.join('experiment/discrete_cql_20250827_072052', 'alpha0.0_bins10')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Train model with 100 epochs
    history = train_single_cql(
        data=data,
        alpha=0.0,
        vp2_bins=10,
        epochs=100,  # Full training
        batch_size=128,
        save_dir=exp_dir
    )
    
    # Save results
    result = {
        'alpha': 0.0,
        'vp2_bins': 10,
        'best_val_loss': history['best_val_loss'],
        'final_train_q_loss': history['train_q_loss'][-1] if history['train_q_loss'] else None,
        'final_train_cql_loss': history['train_cql_loss'][-1] if history['train_cql_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'completed': datetime.now().isoformat()
    }
    
    result_file = os.path.join(exp_dir, 'experiment_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Experiment completed: alpha=0.0, bins=10")
    print(f"   Best val loss: {history['best_val_loss']:.4f}")
    
    return result

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR in experiment alpha=0.0, bins=10: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
