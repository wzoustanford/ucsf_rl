#!/bin/bash
# Training script for LSTM Block Discrete CQL with multiple VP2 bins and alpha values
# Runs 2 jobs in parallel, waits 20 minutes between batches

# Create logs directory
mkdir -p logs
mkdir -p experiment

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "======================================================================" 
echo " STARTING LSTM BLOCK DISCRETE CQL TRAINING - PARALLEL EXECUTION"
echo " Timestamp: $TIMESTAMP"
echo " VP2 bins: 3, 5, 10"
echo " Alphas: 0.00, 0.000001, 0.00001, 0.0001"
echo " Strategy: 2 jobs in parallel, 20 minutes wait between batches"
echo "======================================================================"

# Function to train a single configuration
train_lstm() {
    local alpha=$1
    local vp2_bins=$2
    local log_file="logs/lstm_cql_alpha${alpha}_bins${vp2_bins}_${TIMESTAMP}.log"
    
    echo "[$(date)] Starting training for alpha=$alpha, vp2_bins=$vp2_bins"
    echo "[$(date)] Log file: $log_file"
    
    # Create Python script for this specific configuration
    python3 -c "
import sys
import os
sys.path.append('/home/ubuntu/code/ucsf_rl')

# Disable output buffering
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

from run_lstm_block_discrete_cql_with_logging import train_lstm_block_discrete_cql
import time

print('='*70)
print(f' TRAINING LSTM BLOCK DISCRETE CQL')
print(f' Alpha={$alpha}, VP2 bins={$vp2_bins}')
print('='*70)
print(f'Starting at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('')

try:
    agent, train_buffer, val_buffer, history = train_lstm_block_discrete_cql(
        alpha=$alpha,
        vp2_bins=$vp2_bins,
        sequence_length=20,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        gamma=0.95,
        tau=0.8,
        grad_clip=1.0
    )
    print(f'\\nCompleted at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
    print('✅ Training completed successfully!')
except Exception as e:
    print(f'\\n❌ Training failed with error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" > "$log_file" 2>&1 &
    
    # Store and return the PID
    local pid=$!
    echo $pid > /tmp/lstm_train_pid_${alpha}_${vp2_bins}.txt
    echo $pid
}

# Define all configurations (12 total = 3 bins × 4 alphas)
CONFIGS=(
    "0.00 3"
    "0.00 5"
    "0.00 10"
    "0.000001 3"
    "0.000001 5"
    "0.000001 10"
    "0.00001 3"
    "0.00001 5"
    "0.00001 10"
    "0.0001 3"
    "0.0001 5"
    "0.0001 10"
)

# Process configs in batches of 2
BATCH_NUM=1
for ((i=0; i<${#CONFIGS[@]}; i+=2)); do
    echo ""
    echo "======================================================================"
    echo " BATCH $BATCH_NUM: Starting at $(date)"
    echo "======================================================================"
    
    # Parse first config
    CONFIG1=(${CONFIGS[$i]})
    ALPHA1=${CONFIG1[0]}
    BINS1=${CONFIG1[1]}
    
    # Start first job
    PID1=$(train_lstm $ALPHA1 $BINS1)
    echo "[$(date)] Started: PID $PID1 - alpha=$ALPHA1, bins=$BINS1"
    
    # Check if there's a second config in this batch
    if [ $((i+1)) -lt ${#CONFIGS[@]} ]; then
        CONFIG2=(${CONFIGS[$((i+1))]})
        ALPHA2=${CONFIG2[0]}
        BINS2=${CONFIG2[1]}
        
        # Start second job
        PID2=$(train_lstm $ALPHA2 $BINS2)
        echo "[$(date)] Started: PID $PID2 - alpha=$ALPHA2, bins=$BINS2"
        
        echo ""
        echo "Waiting for batch $BATCH_NUM to complete..."
        
        # Wait for both jobs
        wait $PID1
        STATUS1=$?
        wait $PID2
        STATUS2=$?
        
        if [ $STATUS1 -eq 0 ]; then
            echo "[$(date)] ✅ Alpha=$ALPHA1, bins=$BINS1 completed successfully"
        else
            echo "[$(date)] ❌ Alpha=$ALPHA1, bins=$BINS1 failed"
        fi
        
        if [ $STATUS2 -eq 0 ]; then
            echo "[$(date)] ✅ Alpha=$ALPHA2, bins=$BINS2 completed successfully"
        else
            echo "[$(date)] ❌ Alpha=$ALPHA2, bins=$BINS2 failed"
        fi
    else
        # Only one job in this batch
        echo ""
        echo "Waiting for final job to complete..."
        wait $PID1
        STATUS1=$?
        
        if [ $STATUS1 -eq 0 ]; then
            echo "[$(date)] ✅ Alpha=$ALPHA1, bins=$BINS1 completed successfully"
        else
            echo "[$(date)] ❌ Alpha=$ALPHA1, bins=$BINS1 failed"
        fi
    fi
    
    # If not the last batch, wait 20 minutes
    if [ $((i+2)) -lt ${#CONFIGS[@]} ]; then
        echo ""
        echo "======================================================================"
        echo " BATCH $BATCH_NUM COMPLETE - Waiting 20 minutes before next batch"
        echo " Current time: $(date)"
        echo " Next batch will start at: $(date -d '+20 minutes')"
        echo "======================================================================"
        echo ""
        sleep 1200  # 20 minutes = 1200 seconds
    fi
    
    BATCH_NUM=$((BATCH_NUM + 1))
done

echo ""
echo "======================================================================"
echo " ALL LSTM TRAINING COMPLETED"
echo " Timestamp: $(date)"
echo "======================================================================"
echo ""

echo "Models saved in experiment/ directory:"
for alpha in 0.00 0.000001 0.00001 0.0001; do
    for bins in 3 5 10; do
        # Format alpha for filename
        if [ "$alpha" = "0.00" ]; then
            alpha_str="0.0000"
        elif [ "$alpha" = "0.000001" ]; then
            alpha_str="0.0000"
        elif [ "$alpha" = "0.00001" ]; then
            alpha_str="0.0000"
        elif [ "$alpha" = "0.0001" ]; then
            alpha_str="0.0001"
        else
            alpha_str=$(printf "%.4f" $alpha)
        fi
        
        ls -lh experiment/lstm_block_discrete_cql_alpha${alpha_str}_bins${bins}_*.pt 2>/dev/null
    done
done

echo ""
echo "Training logs saved in logs/ directory:"
ls -la logs/lstm_cql_*${TIMESTAMP}*.log 2>/dev/null

echo ""
echo "Summary of training results:"
for alpha in 0.00 0.000001 0.00001 0.0001; do
    for bins in 3 5 10; do
        log_file="logs/lstm_cql_alpha${alpha}_bins${bins}_${TIMESTAMP}.log"
        if [ -f "$log_file" ]; then
            if grep -q "✅ Training completed successfully" "$log_file"; then
                echo "  ✅ Alpha=$alpha, bins=$bins: SUCCESS"
            else
                echo "  ❌ Alpha=$alpha, bins=$bins: FAILED (check $log_file)"
            fi
        else
            echo "  ⚠️  Alpha=$alpha, bins=$bins: Log file not found"
        fi
    done
done