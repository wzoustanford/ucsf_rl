#!/usr/bin/env python3
"""
Monitor script for discrete CQL experiments
Usage: python3 monitor_discrete_cql.py [log_dir]
"""

import sys
import os
import json
import time
import psutil
from pathlib import Path
from datetime import datetime
import subprocess


def check_process_status(pid):
    """Check if a process is still running"""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def get_file_tail(filepath, lines=10):
    """Get last n lines of a file"""
    try:
        if not os.path.exists(filepath):
            return "File not found"
        
        with open(filepath, 'r') as f:
            content = f.readlines()
            if not content:
                return "File is empty"
            return ''.join(content[-lines:])
    except Exception as e:
        return f"Error reading file: {e}"


def format_time_elapsed(start_time):
    """Format elapsed time"""
    if isinstance(start_time, str):
        start = datetime.fromisoformat(start_time)
    else:
        start = start_time
    
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60
    seconds = elapsed.seconds % 60
    
    if elapsed.days > 0:
        return f"{elapsed.days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"


def check_experiment_completion(exp_dir, exp_info):
    """Check if an experiment has completed by looking for result file"""
    result_file = Path(exp_dir) / f"alpha{exp_info['alpha']}_bins{exp_info['vp2_bins']}" / "experiment_result.json"
    
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                return True, result
        except:
            pass
    
    # Also check for final model file
    final_model = Path(exp_dir) / f"alpha{exp_info['alpha']}_bins{exp_info['vp2_bins']}" / f"cql_discrete_alpha{exp_info['alpha']}_bins{exp_info['vp2_bins']}_final.pt"
    if final_model.exists():
        return True, {"completed": True, "final_model": str(final_model)}
    
    return False, None


def monitor_experiments(log_dir):
    """Monitor running experiments"""
    log_dir = Path(log_dir)
    config_file = log_dir / 'master_config.json'
    
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("DISCRETE CQL EXPERIMENT MONITOR")
    print("="*80)
    print(f"Experiment: {config['timestamp']}")
    print(f"Total experiments: {config['total_experiments']}")
    print(f"Log directory: {log_dir}")
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("="*80)
        print(f"DISCRETE CQL MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        completed = 0
        running = 0
        failed = 0
        
        for exp in config['experiments']:
            # Check process status
            is_running = check_process_status(exp['pid'])
            
            # Check completion
            is_completed, result = check_experiment_completion(config['experiment_dir'], exp)
            
            if is_completed:
                status = "✅ COMPLETED"
                completed += 1
            elif is_running:
                status = "🔄 RUNNING"
                running += 1
            else:
                # Process not running and not completed - likely failed
                status = "❌ FAILED/STOPPED"
                failed += 1
            
            print(f"\n[{exp['id']}/{config['total_experiments']}] Alpha={exp['alpha']}, Bins={exp['vp2_bins']}")
            print(f"  Status: {status}")
            print(f"  PID: {exp['pid']} {'(alive)' if is_running else '(dead)'}")
            print(f"  Elapsed: {format_time_elapsed(exp['started'])}")
            
            # Show last log lines if running or failed
            if is_running or (not is_completed and not is_running):
                log_tail = get_file_tail(exp['log_file'], 3)
                if "Epoch" in log_tail or "Loss" in log_tail:
                    print(f"  Last output: {log_tail.strip()}")
                
                # Check for errors
                err_tail = get_file_tail(exp['err_file'], 2)
                if err_tail and err_tail != "File is empty":
                    print(f"  ⚠️  Error: {err_tail.strip()}")
            
            # Show result if completed
            if is_completed and isinstance(result, dict) and 'best_val_loss' in result:
                print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        
        print("\n" + "="*80)
        print(f"SUMMARY: {completed}/{config['total_experiments']} completed, "
              f"{running} running, {failed} failed")
        print("="*80)
        
        if completed == config['total_experiments']:
            print("\n✅ All experiments completed!")
            break
        
        print("\nPress Ctrl+C to exit monitor (experiments will continue running)")
        print("Refreshing in 30 seconds...")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped. Experiments continue running in background.")
            print(f"To resume monitoring: python3 monitor_discrete_cql.py {log_dir}")
            break


def show_summary(log_dir):
    """Show final summary of all experiments"""
    log_dir = Path(log_dir)
    config_file = log_dir / 'master_config.json'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    results = []
    for exp in config['experiments']:
        is_completed, result = check_experiment_completion(config['experiment_dir'], exp)
        if is_completed and isinstance(result, dict) and 'best_val_loss' in result:
            results.append({
                'alpha': exp['alpha'],
                'bins': exp['vp2_bins'],
                'best_val_loss': result['best_val_loss']
            })
    
    if results:
        # Sort by best validation loss
        results.sort(key=lambda x: x['best_val_loss'])
        
        print("\nTop 5 configurations:")
        for i, r in enumerate(results[:5], 1):
            print(f"{i}. Alpha={r['alpha']}, Bins={r['bins']}: Val Loss={r['best_val_loss']:.4f}")
        
        # Find best for each bin size
        print("\nBest configuration per bin size:")
        for bins in config['vp2_bins_options']:
            bin_results = [r for r in results if r['bins'] == bins]
            if bin_results:
                best = min(bin_results, key=lambda x: x['best_val_loss'])
                print(f"  Bins={bins}: Alpha={best['alpha']}, Val Loss={best['best_val_loss']:.4f}")


def main():
    if len(sys.argv) < 2:
        # Try to find most recent log directory
        log_parent = Path('logs')
        if log_parent.exists():
            log_dirs = sorted([d for d in log_parent.iterdir() if d.is_dir() and 'discrete_cql' in d.name])
            if log_dirs:
                log_dir = log_dirs[-1]
                print(f"Using most recent log directory: {log_dir}")
            else:
                print("No log directories found. Please specify log directory.")
                print("Usage: python3 monitor_discrete_cql.py <log_dir>")
                sys.exit(1)
        else:
            print("No logs directory found. Please specify log directory.")
            print("Usage: python3 monitor_discrete_cql.py <log_dir>")
            sys.exit(1)
    else:
        log_dir = sys.argv[1]
    
    if not Path(log_dir).exists():
        print(f"Error: Log directory {log_dir} does not exist")
        sys.exit(1)
    
    try:
        monitor_experiments(log_dir)
        show_summary(log_dir)
    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()