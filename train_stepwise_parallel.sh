#!/bin/bash
# Training script for Stepwise CQL with multiple alpha values
# Runs 2 jobs in parallel, waits 1 hour between batches

# Create logs directory
mkdir -p logs
mkdir -p experiment

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "======================================================================" 
echo " STARTING STEPWISE CQL TRAINING - PARALLEL EXECUTION"
echo " Timestamp: $TIMESTAMP"
echo " Alphas: 0.00, 0.000001, 0.00001, 0.0001"
echo " Epochs: 200 per alpha"
echo " Strategy: 2 jobs in parallel, 1 hour wait between batches"
echo "======================================================================"

# Function to train a single alpha
train_alpha() {
    local alpha=$1
    local log_file="logs/stepwise_cql_alpha${alpha}_${TIMESTAMP}.log"
    
    echo "[$(date)] Starting training for alpha=$alpha"
    echo "[$(date)] Log file: $log_file"
    
    # Create Python script for this specific alpha
    python3 -c "
import sys
sys.path.append('/home/ubuntu/code/ucsf_rl')
from run_unified_stepwise_cql_allalphas import train_unified_stepwise_cql
import time

print('='*70)
print(f' TRAINING STEPWISE CQL WITH ALPHA={$alpha}')
print('='*70)
print(f'Starting at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('')

try:
    agent, pipeline = train_unified_stepwise_cql(alpha=$alpha, max_step=0.1)
    print(f'\\nCompleted at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
    print('✅ Training completed successfully!')
except Exception as e:
    print(f'\\n❌ Training failed with error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" > "$log_file" 2>&1 &
    
    # Store the PID in a temp file and return it
    local pid=$!
    echo $pid > /tmp/train_pid_${alpha}.txt
    echo $pid
}

# Alpha values to train
ALPHAS=(0.00 0.000001 0.00001 0.0001)

# Process alphas in batches of 2
echo ""
echo "======================================================================"
echo " BATCH 1: alpha=0.00 and alpha=0.000001"
echo "======================================================================"

# Start first batch (2 jobs in parallel)
PID1=$(train_alpha 0.00)
PID2=$(train_alpha 0.000001)

echo "[$(date)] Started 2 training jobs:"
echo "  - PID $PID1: alpha=0.00"
echo "  - PID $PID2: alpha=0.000001"
echo ""
echo "Waiting for batch 1 to complete..."

# Wait for both jobs to complete
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?

if [ $STATUS1 -eq 0 ]; then
    echo "[$(date)] ✅ Alpha=0.00 completed successfully"
else
    echo "[$(date)] ❌ Alpha=0.00 failed"
fi

if [ $STATUS2 -eq 0 ]; then
    echo "[$(date)] ✅ Alpha=0.000001 completed successfully"
else
    echo "[$(date)] ❌ Alpha=0.000001 failed"
fi

echo ""
echo "======================================================================"
echo " BATCH 1 COMPLETE - Waiting 1 hour before starting batch 2"
echo " Current time: $(date)"
echo " Next batch will start at: $(date -d '+1 hour')"
echo "======================================================================"
echo ""

# Wait 1 hour (3600 seconds)
sleep 3600

echo ""
echo "======================================================================"
echo " BATCH 2: alpha=0.00001 and alpha=0.0001"
echo "======================================================================"

# Start second batch (2 jobs in parallel)
PID3=$(train_alpha 0.00001)
PID4=$(train_alpha 0.0001)

echo "[$(date)] Started 2 training jobs:"
echo "  - PID $PID3: alpha=0.00001"
echo "  - PID $PID4: alpha=0.0001"
echo ""
echo "Waiting for batch 2 to complete..."

# Wait for both jobs to complete
wait $PID3
STATUS3=$?
wait $PID4
STATUS4=$?

if [ $STATUS3 -eq 0 ]; then
    echo "[$(date)] ✅ Alpha=0.00001 completed successfully"
else
    echo "[$(date)] ❌ Alpha=0.00001 failed"
fi

if [ $STATUS4 -eq 0 ]; then
    echo "[$(date)] ✅ Alpha=0.0001 completed successfully"
else
    echo "[$(date)] ❌ Alpha=0.0001 failed"
fi

echo ""
echo "======================================================================"
echo " ALL TRAINING COMPLETED"
echo " Timestamp: $(date)"
echo "======================================================================"
echo ""
echo "Models saved in experiment/ directory:"
ls -la experiment/stepwise_cql_*.pt 2>/dev/null | tail -20

echo ""
echo "Training logs saved in logs/ directory:"
ls -la logs/stepwise_cql_*${TIMESTAMP}*.log 2>/dev/null

echo ""
echo "Summary of training results:"
for alpha in 0.00 0.000001 0.00001 0.0001; do
    log_file="logs/stepwise_cql_alpha${alpha}_${TIMESTAMP}.log"
    if [ -f "$log_file" ]; then
        if grep -q "✅ Training completed successfully" "$log_file"; then
            echo "  ✅ Alpha=$alpha: SUCCESS"
        else
            echo "  ❌ Alpha=$alpha: FAILED (check $log_file)"
        fi
    else
        echo "  ⚠️  Alpha=$alpha: Log file not found"
    fi
done

echo ""
echo "To monitor logs in real-time, use:"
echo "  tail -f logs/stepwise_cql_alpha<alpha>_${TIMESTAMP}.log"