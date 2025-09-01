#!/bin/bash

# Launch all discrete CQL experiments
# This script runs all combinations of alpha and VP2 bins in background
# Runs 4 jobs at a time to avoid GPU memory issues

echo "========================================"
echo "LAUNCHING ALL DISCRETE CQL EXPERIMENTS"
echo "========================================"

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/discrete_cql_${TIMESTAMP}"
EXP_DIR="experiment/discrete_cql_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$EXP_DIR"

echo "Log directory: $LOG_DIR"
echo "Experiment directory: $EXP_DIR"
echo ""

# Alpha values and VP2 bins to test
ALPHAS=(0.0 0.001 0.01)
VP2_BINS=(5 10 30 50)

# PID tracking file
PID_FILE="$LOG_DIR/running_pids.txt"
> "$PID_FILE"

# Status file
STATUS_FILE="$LOG_DIR/experiment_status.txt"
> "$STATUS_FILE"

# Maximum concurrent jobs
MAX_JOBS=4

# Function to count running jobs
count_running_jobs() {
    local count=0
    while IFS= read -r line; do
        pid=$(echo $line | awk '{print $1}')
        if kill -0 $pid 2>/dev/null; then
            count=$((count + 1))
        fi
    done < "$PID_FILE"
    echo $count
}

# Function to wait for a slot to become available
wait_for_slot() {
    while [ $(count_running_jobs) -ge $MAX_JOBS ]; do
        echo "  Waiting for a slot (currently $(count_running_jobs)/$MAX_JOBS jobs running)..."
        sleep 10
    done
}

# Launch each experiment
COUNTER=0
TOTAL=$((${#ALPHAS[@]} * ${#VP2_BINS[@]}))

echo "Running experiments with max $MAX_JOBS concurrent jobs"
echo ""

for BINS in "${VP2_BINS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        COUNTER=$((COUNTER + 1))
        
        # Wait for a slot if needed
        wait_for_slot
        
        echo "[$COUNTER/$TOTAL] Launching alpha=$ALPHA, bins=$BINS"
        
        # Create Python command that directly uses the module
        PYTHON_CMD="
import sys
import os
import torch

# Set CUDA device management
if torch.cuda.is_available():
    # Enable GPU memory growth to avoid pre-allocating all memory
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

sys.path.insert(0, '/home/ubuntu/code/ucsf_rl')
from run_unified_dual_cql_block_discrete import prepare_data, train_single_cql
import json
from datetime import datetime

print('='*70)
print(f'Starting experiment: alpha=$ALPHA, VP2_bins=$BINS')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print('='*70)

try:
    # Load data
    data = prepare_data('sample_data_oviss.csv', $BINS)
    
    # Create experiment directory
    exp_dir = os.path.join('$EXP_DIR', 'alpha${ALPHA}_bins${BINS}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Train model
    history = train_single_cql(
        data=data,
        alpha=$ALPHA,
        vp2_bins=$BINS,
        epochs=100,
        batch_size=128,
        save_dir=exp_dir
    )
    
    # Save results
    result = {
        'alpha': $ALPHA,
        'vp2_bins': $BINS,
        'best_val_loss': history['best_val_loss'],
        'final_train_q_loss': history['train_q_loss'][-1] if history['train_q_loss'] else None,
        'final_train_cql_loss': history['train_cql_loss'][-1] if history['train_cql_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'completed': datetime.now().isoformat()
    }
    
    with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    print('='*70)
    print(f'✅ Completed: alpha=$ALPHA, bins=$BINS')
    print(f'   Best val loss: {history[\"best_val_loss\"]:.4f}')
    print('='*70)
    
    # Clear GPU cache after completion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f'❌ Error in experiment alpha=$ALPHA, bins=$BINS: {e}')
    import traceback
    traceback.print_exc()
    
    # Save error info
    error_info = {
        'alpha': $ALPHA,
        'vp2_bins': $BINS,
        'error': str(e),
        'failed': datetime.now().isoformat()
    }
    
    exp_dir = os.path.join('$EXP_DIR', 'alpha${ALPHA}_bins${BINS}')
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'error.json'), 'w') as f:
        json.dump(error_info, f, indent=2)
"
        
        # Launch in background
        nohup python3 -c "$PYTHON_CMD" \
            > "$LOG_DIR/alpha${ALPHA}_bins${BINS}.log" \
            2> "$LOG_DIR/alpha${ALPHA}_bins${BINS}.err" &
        
        PID=$!
        echo "  Started with PID: $PID"
        echo "$PID alpha${ALPHA}_bins${BINS}" >> "$PID_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Started alpha${ALPHA}_bins${BINS} (PID: $PID)" >> "$STATUS_FILE"
        
        # Small delay before checking next
        sleep 3
    done
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for all jobs to complete..."

while [ $(count_running_jobs) -gt 0 ]; do
    running=$(count_running_jobs)
    echo "  $running jobs still running..."
    sleep 30
done

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================"
echo ""
echo "Results in: $EXP_DIR"
echo "Logs in: $LOG_DIR"
echo ""

# Show summary of results
echo "Summary of results:"
for BINS in "${VP2_BINS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        RESULT_FILE="$EXP_DIR/alpha${ALPHA}_bins${BINS}/result.json"
        ERROR_FILE="$EXP_DIR/alpha${ALPHA}_bins${BINS}/error.json"
        
        if [ -f "$RESULT_FILE" ]; then
            VAL_LOSS=$(python3 -c "import json; print(json.load(open('$RESULT_FILE'))['best_val_loss'])" 2>/dev/null || echo "N/A")
            echo "  Alpha=$ALPHA, Bins=$BINS: Val Loss=$VAL_LOSS ✅"
        elif [ -f "$ERROR_FILE" ]; then
            echo "  Alpha=$ALPHA, Bins=$BINS: FAILED ❌"
        else
            echo "  Alpha=$ALPHA, Bins=$BINS: UNKNOWN ⚠️"
        fi
    done
done