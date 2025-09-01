#!/bin/bash

# LSTM Batched Training Script
# Runs LSTM experiments with bins=10 and alpha in {0.0, 0.0001}
# Submits 2 jobs at a time with 40-minute delays between batches

# Configuration
ALPHAS=(0.0001 0.001)
BINS=10
EPOCHS=100

# Create directories
mkdir -p logs
mkdir -p experiment

echo "=========================================="
echo " LSTM BATCHED TRAINING"
echo "=========================================="
echo "Configuration:"
echo "  Bins: $BINS"
echo "  Alphas: ${ALPHAS[@]}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: 2 jobs at a time"
echo "  Delay between batches: 40 minutes"
echo "=========================================="

# Function to run a single experiment
run_experiment() {
    local alpha=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/lstm_alpha${alpha}_bins${BINS}_${timestamp}.log"
    
    echo "[$(date)] Starting LSTM training: alpha=$alpha, bins=$BINS"
    echo "  Log file: $log_file"
    
    nohup python3 run_lstm_block_discrete_cql_with_args.py \
        --alpha $alpha \
        --vp2_bins $BINS \
        --epochs $EPOCHS \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "  Started with PID: $pid"
    
    # Save experiment info
    echo "{\"pid\": $pid, \"alpha\": $alpha, \"bins\": $BINS, \"start_time\": \"$(date)\", \"log_file\": \"$log_file\"}" \
        > "logs/lstm_${pid}.json"
}

# Main execution
echo ""
echo "Starting training at $(date)"
echo ""

# Since we only have 2 experiments total, run them both
run_experiment ${ALPHAS[0]}
sleep 5
run_experiment ${ALPHAS[1]}

echo ""
echo "=========================================="
echo " ALL EXPERIMENTS LAUNCHED"
echo "=========================================="
echo "Monitor progress with: ./check_lstm_status.sh"
echo "View logs with: tail -f logs/lstm_*.log"
echo ""