#!/usr/bin/env python3
"""
Monitor Dual Mixed CQL training progress
"""

import os
import time
import re
from datetime import datetime


def check_process(pid):
    """Check if process is running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def parse_log_file(log_path):
    """Parse training log to extract progress"""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all alpha training sections
    alpha_pattern = r'DUAL MIXED CQL TRAINING WITH ALPHA=([\d.]+)'
    alpha_matches = re.findall(alpha_pattern, content)
    
    # Find all epoch progress lines
    epoch_pattern = r'Epoch (\d+): Q1 Loss=([\d.]+), CQL1 Loss=([\d.]+), Val Q-value=([-\d.]+), Time=([\d.]+)min'
    epoch_matches = re.findall(epoch_pattern, content)
    
    # Find completion messages
    completion_pattern = r'Dual Mixed CQL \(alpha=([\d.]+)\) training completed in ([\d.]+) minutes'
    completion_matches = re.findall(completion_pattern, content)
    
    return {
        'alphas_started': alpha_matches,
        'epochs': epoch_matches,
        'completed': completion_matches
    }


def main():
    """Monitor training progress"""
    print("="*70)
    print(f" DUAL MIXED CQL TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if process is running
    pid_file = '/home/ubuntu/code/ucsf_rl/dual_mixed_training.pid'
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        is_running = check_process(pid)
        print(f"\nProcess PID: {pid}")
        print(f"Status: {'🏃 Running' if is_running else '✅ Completed'}")
    else:
        print("\nNo training process found.")
        pid = None
        is_running = False
    
    # Parse log file
    log_path = '/home/ubuntu/code/ucsf_rl/logs/dual_mixed_cql_training.out'
    progress = parse_log_file(log_path)
    
    if progress:
        print("\n" + "-"*70)
        print("PROGRESS:")
        print("-"*70)
        
        # Show which alphas have started
        if progress['alphas_started']:
            print("\n📋 Training Sessions Started:")
            for alpha in progress['alphas_started']:
                status = "✅" if any(c[0] == alpha for c in progress['completed']) else "🏃"
                print(f"  {status} Alpha = {alpha}")
        
        # Show latest epoch if training
        if progress['epochs'] and is_running:
            latest_epoch = progress['epochs'][-1]
            epoch_num, q1_loss, cql_loss, val_q, time_min = latest_epoch
            print(f"\n🏃 Current Progress:")
            print(f"  Epoch: {epoch_num}/100")
            print(f"  Q1 Loss: {float(q1_loss):.4f}")
            print(f"  CQL Loss: {float(cql_loss):.4f}")
            print(f"  Val Q-value: {float(val_q):.4f}")
            print(f"  Time: {float(time_min):.1f} min")
        
        # Show completed models
        if progress['completed']:
            print(f"\n✅ Completed Models:")
            for alpha, time_min in progress['completed']:
                print(f"  Alpha {alpha}: {float(time_min):.1f} minutes")
    
    # Check saved models
    print("\n" + "-"*70)
    print("SAVED MODELS:")
    print("-"*70)
    
    experiment_dir = '/home/ubuntu/code/ucsf_rl/experiment'
    alphas = ['0.0000', '0.0010', '0.0100']
    
    for alpha in alphas:
        best_model = f'{experiment_dir}/dual_rev_cql_alpha{alpha}_best.pt'
        final_model = f'{experiment_dir}/dual_rev_cql_alpha{alpha}_final.pt'
        
        if os.path.exists(best_model) or os.path.exists(final_model):
            status = "✅"
            models = []
            if os.path.exists(best_model):
                models.append("best")
            if os.path.exists(final_model):
                models.append("final")
            print(f"  {status} Alpha {alpha}: {', '.join(models)}")
        else:
            print(f"  ⏳ Alpha {alpha}: not yet saved")
    
    # Check log file sizes
    print("\n" + "-"*70)
    print("LOG FILES:")
    print("-"*70)
    
    if os.path.exists(log_path):
        size = os.path.getsize(log_path) / 1024
        print(f"Output log: {size:.1f} KB")
    
    err_path = '/home/ubuntu/code/ucsf_rl/logs/dual_mixed_cql_training.err'
    if os.path.exists(err_path):
        err_size = os.path.getsize(err_path)
        if err_size > 0:
            print(f"⚠️  Error log: {err_size} bytes")
            with open(err_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"   Last error: {lines[-1].strip()[:100]}...")
        else:
            print("✅ No errors")
    
    print("\n" + "-"*70)
    print("Commands:")
    print("  • Watch output: tail -f /home/ubuntu/code/ucsf_rl/logs/dual_mixed_cql_training.out")
    print("  • Check errors: cat /home/ubuntu/code/ucsf_rl/logs/dual_mixed_cql_training.err")
    print(f"  • Kill process: kill {pid}" if pid else "")
    
    print(f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()