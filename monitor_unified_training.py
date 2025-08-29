#!/usr/bin/env python3
"""
Monitor progress of unified CQL training.
"""

import os
import time
import re
from datetime import datetime

def check_process(pid):
    """Check if process is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def parse_log_file(log_path):
    """Parse training log to extract progress."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find current model being trained
    current_model = None
    current_alpha = None
    current_epoch = None
    latest_metrics = {}
    
    for line in reversed(lines):
        # Check for model type
        if "Training BINARY CQL with alpha=" in line:
            match = re.search(r'alpha=([\d.]+)', line)
            if match:
                current_model = "Binary"
                current_alpha = float(match.group(1))
        elif "Training DUAL CQL with alpha=" in line:
            match = re.search(r'alpha=([\d.]+)', line)
            if match:
                current_model = "Dual"
                current_alpha = float(match.group(1))
        
        # Check for epoch progress
        if "Epoch " in line and "Q1 Loss=" in line:
            match = re.search(r'Epoch (\d+):', line)
            if match:
                current_epoch = int(match.group(1))
                
                # Extract metrics
                q1_match = re.search(r'Q1 Loss=([\d.]+)', line)
                cql_match = re.search(r'CQL Loss=([\d.]+)', line)
                val_match = re.search(r'Val Q=([-\d.]+)', line)
                
                if q1_match:
                    latest_metrics['q1_loss'] = float(q1_match.group(1))
                if cql_match:
                    latest_metrics['cql_loss'] = float(cql_match.group(1))
                if val_match:
                    latest_metrics['val_q'] = float(val_match.group(1))
                break
    
    # Check for completion
    completed_models = []
    for line in lines:
        if "completed in" in line.lower():
            if "BINARY CQL" in line:
                alpha_match = re.search(r'alpha=([\d.]+)', line)
                if alpha_match:
                    completed_models.append(f"Binary α={alpha_match.group(1)}")
            elif "DUAL CQL" in line:
                alpha_match = re.search(r'alpha=([\d.]+)', line)
                if alpha_match:
                    completed_models.append(f"Dual α={alpha_match.group(1)}")
    
    return {
        'current_model': current_model,
        'current_alpha': current_alpha,
        'current_epoch': current_epoch,
        'metrics': latest_metrics,
        'completed': completed_models
    }

def main():
    """Monitor training progress."""
    print("="*70)
    print(f" UNIFIED CQL TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if process is running
    if os.path.exists('/home/ubuntu/code/ucsf_rl/training.pid'):
        with open('/home/ubuntu/code/ucsf_rl/training.pid', 'r') as f:
            pid = int(f.read().strip())
        
        is_running = check_process(pid)
        print(f"\nProcess PID: {pid}")
        print(f"Status: {'🏃 Running' if is_running else '✅ Completed'}")
    else:
        print("\nNo training process found.")
        pid = None
        is_running = False
    
    # Parse log file
    log_path = '/home/ubuntu/code/ucsf_rl/logs/unified_cql_training_direct.out'
    progress = parse_log_file(log_path)
    
    if progress:
        print("\n" + "-"*70)
        print("PROGRESS:")
        print("-"*70)
        
        # Completed models
        if progress['completed']:
            print("\n✅ Completed Models:")
            for model in progress['completed']:
                print(f"   • {model}")
        
        # Current training
        if progress['current_model'] and progress['current_epoch']:
            print(f"\n🏃 Currently Training:")
            print(f"   Model: {progress['current_model']} CQL")
            print(f"   Alpha: {progress['current_alpha']}")
            print(f"   Epoch: {progress['current_epoch']}/100")
            
            if progress['metrics']:
                print(f"\n📊 Latest Metrics:")
                if 'q1_loss' in progress['metrics']:
                    print(f"   Q1 Loss: {progress['metrics']['q1_loss']:.4f}")
                if 'cql_loss' in progress['metrics']:
                    print(f"   CQL Loss: {progress['metrics']['cql_loss']:.4f}")
                if 'val_q' in progress['metrics']:
                    print(f"   Val Q: {progress['metrics']['val_q']:.4f}")
    
    # Check log file sizes
    print("\n" + "-"*70)
    print("LOG FILES:")
    print("-"*70)
    
    if os.path.exists(log_path):
        size = os.path.getsize(log_path) / 1024
        print(f"Output log: {size:.1f} KB")
    
    err_path = '/home/ubuntu/code/ucsf_rl/logs/unified_cql_training_direct.err'
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
    
    # Expected models to train
    print("\n" + "-"*70)
    print("EXPECTED TRAINING SEQUENCE:")
    print("-"*70)
    models = [
        "Binary CQL α=0.0",
        "Binary CQL α=0.001", 
        "Binary CQL α=0.01",
        "Dual CQL α=0.0",
        "Dual CQL α=0.001",
        "Dual CQL α=0.01"
    ]
    
    completed = progress['completed'] if progress else []
    for model in models:
        status = "✅" if model in completed else "⏳"
        print(f"  {status} {model}")
    
    print("\n" + "-"*70)
    print("Commands:")
    print("  • Watch output: tail -f /home/ubuntu/code/ucsf_rl/logs/unified_cql_training_direct.out")
    print("  • Check errors: cat /home/ubuntu/code/ucsf_rl/logs/unified_cql_training_direct.err")
    print(f"  • Kill process: kill {pid}" if pid else "")
    
    print(f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()