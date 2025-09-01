#!/bin/bash

# LSTM Block Discrete CQL Training Script
# ========================================
# Trains all combinations of vp2_bins and alphas
# Runs 2 jobs concurrently with 20-minute pause between batches

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create directories
mkdir -p logs
mkdir -p experiment

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/lstm_block_discrete_master_${TIMESTAMP}.log"

# Concurrent jobs limit
MAX_JOBS=2
PAUSE_MINUTES=20

# Print configuration
echo "======================================================================" | tee -a "$MASTER_LOG"
echo " LSTM BLOCK DISCRETE CQL - FULL TRAINING SUITE" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Configurations to train:" | tee -a "$MASTER_LOG"
echo "  VP2 bins: 3, 5, 10" | tee -a "$MASTER_LOG"
echo "  Alphas: 0.0, 0.00001, 0.0001, 0.001" | tee -a "$MASTER_LOG"
echo "  Epochs per config: 100" | tee -a "$MASTER_LOG"
echo "  Concurrent jobs: $MAX_JOBS" | tee -a "$MASTER_LOG"
echo "  Pause between batches: $PAUSE_MINUTES minutes" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Arrays for configurations
VPS_BINS=(3 5 10)
ALPHAS=(0.0 0.00001 0.0001 0.001)

# Build list of all configurations
CONFIGS=()
for vp2_bins in "${VPS_BINS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        CONFIGS+=("${alpha},${vp2_bins}")
    done
done

TOTAL=${#CONFIGS[@]}
SUCCESS=0
FAILED=0

echo "Total configurations: $TOTAL" | tee -a "$MASTER_LOG"
echo "Number of batches: $(( (TOTAL + MAX_JOBS - 1) / MAX_JOBS ))" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Function to train a single configuration
train_config() {
    local alpha=$1
    local vp2_bins=$2
    local config_num=$3
    local total=$4
    
    # Format alpha for display
    alpha_str=$(printf "%.5f" $alpha)
    
    echo "[$(date '+%H:%M:%S')] [$config_num/$total] Starting: alpha=$alpha_str, vp2_bins=$vp2_bins" | tee -a "$MASTER_LOG"
    
    # Create individual log file for this configuration
    # Format alpha to avoid collisions using string comparison
    if [ "$alpha" = "0.0" ]; then
        alpha_file="0.0"
    elif [ "$alpha" = "0.00001" ]; then
        alpha_file="1e-05"
    elif [ "$alpha" = "0.0001" ]; then
        alpha_file="0.0001"
    else
        alpha_file=$(printf "%.4f" $alpha)
    fi
    CONFIG_LOG="logs/lstm_block_discrete_alpha${alpha_file}_bins${vp2_bins}_${TIMESTAMP}.log"
    
    # Create Python script for this specific configuration
    cat > "train_config_${alpha}_${vp2_bins}.py" << EOF
#!/usr/bin/env python3
import sys
sys.path.append('$SCRIPT_DIR')
from run_lstm_block_discrete_cql_with_logging import train_lstm_block_discrete_cql

print("Training LSTM Block Discrete CQL: alpha=$alpha, vp2_bins=$vp2_bins")

try:
    agent, train_buffer, val_buffer, history = train_lstm_block_discrete_cql(
        alpha=$alpha,
        vp2_bins=$vp2_bins,
        sequence_length=20,
        burn_in_length=8,
        overlap=10,
        hidden_dim=64,
        lstm_hidden=64,
        num_lstm_layers=2,
        batch_size=32,
        epochs=100,
        learning_rate=1e-3,
        tau=0.8,  # Aligned with block discrete CQL
        gamma=0.95,
        grad_clip=1.0,
        buffer_capacity=50000,
        save_dir='experiment',
        log_dir='logs',
        log_every=1
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)
EOF
    
    # Run the training in background
    START_TIME=$(date +%s)
    
    (
        if python3 "train_config_${alpha}_${vp2_bins}.py" >> "$CONFIG_LOG" 2>&1; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            ELAPSED_MIN=$((ELAPSED / 60))
            
            echo "[$(date '+%H:%M:%S')] ✅ SUCCESS: alpha=$alpha_str, vp2_bins=$vp2_bins (${ELAPSED_MIN} minutes)" | tee -a "$MASTER_LOG"
            echo "success" > "status_${alpha}_${vp2_bins}.tmp"
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            ELAPSED_MIN=$((ELAPSED / 60))
            
            echo "[$(date '+%H:%M:%S')] ❌ FAILED: alpha=$alpha_str, vp2_bins=$vp2_bins (${ELAPSED_MIN} minutes)" | tee -a "$MASTER_LOG"
            echo "   Check log: $CONFIG_LOG" | tee -a "$MASTER_LOG"
            echo "failed" > "status_${alpha}_${vp2_bins}.tmp"
        fi
        
        # Clean up temp script
        rm -f "train_config_${alpha}_${vp2_bins}.py"
    ) &
    
    # Return the PID
    echo $!
}

# Track start time
SUITE_START=$(date +%s)

# Process configurations in batches
BATCH_NUM=0
for ((i=0; i<$TOTAL; i+=MAX_JOBS)); do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_SIZE=$MAX_JOBS
    
    # Calculate actual batch size (might be less for last batch)
    if [ $((i + MAX_JOBS)) -gt $TOTAL ]; then
        BATCH_SIZE=$((TOTAL - i))
    fi
    
    echo "" | tee -a "$MASTER_LOG"
    echo "======================================================================" | tee -a "$MASTER_LOG"
    echo " BATCH $BATCH_NUM - Starting $BATCH_SIZE jobs" | tee -a "$MASTER_LOG"
    echo "======================================================================" | tee -a "$MASTER_LOG"
    
    # Start jobs in this batch
    PIDS=()
    for ((j=0; j<BATCH_SIZE; j++)); do
        if [ $((i + j)) -lt $TOTAL ]; then
            CONFIG=${CONFIGS[$((i + j))]}
            IFS=',' read -r alpha vp2_bins <<< "$CONFIG"
            CONFIG_NUM=$((i + j + 1))
            
            PID=$(train_config $alpha $vp2_bins $CONFIG_NUM $TOTAL)
            PIDS+=($PID)
        fi
    done
    
    echo "[$(date '+%H:%M:%S')] Batch $BATCH_NUM: Started ${#PIDS[@]} jobs with PIDs: ${PIDS[@]}" | tee -a "$MASTER_LOG"
    
    # Wait for all jobs in this batch to complete
    echo "[$(date '+%H:%M:%S')] Waiting for batch $BATCH_NUM to complete..." | tee -a "$MASTER_LOG"
    
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "[$(date '+%H:%M:%S')] Batch $BATCH_NUM completed" | tee -a "$MASTER_LOG"
    
    # Count successes and failures from status files
    for ((j=0; j<BATCH_SIZE; j++)); do
        if [ $((i + j)) -lt $TOTAL ]; then
            CONFIG=${CONFIGS[$((i + j))]}
            IFS=',' read -r alpha vp2_bins <<< "$CONFIG"
            
            if [ -f "status_${alpha}_${vp2_bins}.tmp" ]; then
                STATUS=$(cat "status_${alpha}_${vp2_bins}.tmp")
                if [ "$STATUS" = "success" ]; then
                    SUCCESS=$((SUCCESS + 1))
                else
                    FAILED=$((FAILED + 1))
                fi
                rm -f "status_${alpha}_${vp2_bins}.tmp"
            fi
        fi
    done
    
    # Progress update
    COMPLETED=$((SUCCESS + FAILED))
    echo "" | tee -a "$MASTER_LOG"
    echo "Overall Progress: $COMPLETED/$TOTAL (✅ $SUCCESS completed, ❌ $FAILED failed)" | tee -a "$MASTER_LOG"
    
    # Estimate remaining time
    if [ $COMPLETED -lt $TOTAL ]; then
        SUITE_NOW=$(date +%s)
        SUITE_ELAPSED=$((SUITE_NOW - SUITE_START))
        AVG_TIME=$((SUITE_ELAPSED / COMPLETED))
        REMAINING=$(((TOTAL - COMPLETED) * AVG_TIME))
        REMAINING_MIN=$((REMAINING / 60))
        REMAINING_HOURS=$((REMAINING / 3600))
        echo "Estimated remaining time: ${REMAINING_MIN} minutes (${REMAINING_HOURS}.$(( (REMAINING_MIN % 60) * 10 / 60 )) hours)" | tee -a "$MASTER_LOG"
    fi
    
    # Pause between batches (except after the last batch)
    if [ $((i + MAX_JOBS)) -lt $TOTAL ]; then
        echo "" | tee -a "$MASTER_LOG"
        echo "[$(date '+%H:%M:%S')] Pausing for $PAUSE_MINUTES minutes before next batch..." | tee -a "$MASTER_LOG"
        echo "Next batch will start at: $(date -d "+$PAUSE_MINUTES minutes" '+%H:%M:%S')" | tee -a "$MASTER_LOG"
        sleep $((PAUSE_MINUTES * 60))
    fi
done

# Final summary
SUITE_END=$(date +%s)
TOTAL_TIME=$((SUITE_END - SUITE_START))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_HOURS=$((TOTAL_TIME / 3600))

echo "" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo " TRAINING COMPLETE" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "Total elapsed time: ${TOTAL_MIN} minutes (${TOTAL_HOURS} hours)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Results:" | tee -a "$MASTER_LOG"
echo "  Completed: $SUCCESS/$TOTAL" | tee -a "$MASTER_LOG"
echo "  Failed: $FAILED/$TOTAL" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

if [ $SUCCESS -gt 0 ]; then
    echo "Trained models saved in experiment/ directory:" | tee -a "$MASTER_LOG"
    for config in "${CONFIGS[@]}"; do
        IFS=',' read -r alpha vp2_bins <<< "$config"
        # Format alpha to match Python naming
        if [ "$alpha" = "0.0" ]; then
            alpha_file="0.0"
        elif [ "$alpha" = "0.00001" ]; then
            alpha_file="1p0em05"  # Matches Python's replace('.', 'p').replace('-', 'm')
        elif [ "$alpha" = "0.0001" ]; then
            alpha_file="0.0001"
        else
            alpha_file=$(printf "%.4f" $alpha)
        fi
        if [ -f "experiment/lstm_block_discrete_cql_alpha${alpha_file}_bins${vp2_bins}_best.pt" ]; then
            echo "  - lstm_block_discrete_cql_alpha${alpha_file}_bins${vp2_bins}_*.pt" | tee -a "$MASTER_LOG"
        fi
    done
fi

echo "" | tee -a "$MASTER_LOG"
echo "Master log saved to: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "Individual logs saved in: logs/" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

# Clean up any remaining status files
rm -f status_*.tmp