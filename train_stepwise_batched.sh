#!/bin/bash

# Train Stepwise CQL models in batches of 2 with 40-minute delays
# This ensures we don't overwhelm the system and can monitor progress

echo "=========================================="
echo "STEPWISE CQL BATCHED TRAINING"
echo "=========================================="
echo "Starting at: $(date)"
echo "Running 100 epochs per experiment"
echo "Submitting in batches of 2 jobs, 40 minutes apart"
echo ""

# Create logs directory
mkdir -p logs
mkdir -p experiment

# Define parameter combinations
ALPHAS=(0.0 0.0001 0.001 0.01)
MAX_STEPS=(0.1 0.2)

# Create all experiment combinations
EXPERIMENTS=()
for alpha in "${ALPHAS[@]}"; do
    for max_step in "${MAX_STEPS[@]}"; do
        EXPERIMENTS+=("$alpha:$max_step")
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments to run: $TOTAL"
echo "Will run in $((($TOTAL + 1) / 2)) batches of 2"
echo ""

# Clear previous running experiments file
> logs/running_experiments.txt
echo "PID | Alpha | MaxStep | StartTime | LogFile" > logs/experiment_tracker.txt

# Function to run single experiment
run_experiment() {
    local alpha=$1
    local max_step=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/stepwise_alpha${alpha}_maxstep${max_step}_${timestamp}.log"
    
    echo "  Starting: alpha=$alpha, max_step=$max_step"
    echo "  Log: $log_file"
    
    # Run training in background with nohup
    nohup python3 run_unified_stepwise_cql_allalphas.py \
        --alpha $alpha \
        --max_step $max_step \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "  PID: $pid"
    
    # Track the experiment
    echo "$pid | $alpha | $max_step | $(date) | $log_file" >> logs/experiment_tracker.txt
    
    return $pid
}

# Function to check if process is still running
is_running() {
    kill -0 $1 2>/dev/null
}

# Run experiments in batches
BATCH_NUM=0
for ((i=0; i<$TOTAL; i+=2)); do
    BATCH_NUM=$((BATCH_NUM + 1))
    
    echo "=========================================="
    echo "BATCH $BATCH_NUM - Starting at $(date)"
    echo "=========================================="
    
    # Start first experiment in batch
    IFS=':' read -r alpha1 max_step1 <<< "${EXPERIMENTS[$i]}"
    echo "[Experiment $((i+1))/$TOTAL]"
    run_experiment $alpha1 $max_step1
    PID1=$!
    
    # Small delay before starting second
    sleep 5
    
    # Start second experiment in batch if it exists
    if [ $((i+1)) -lt $TOTAL ]; then
        IFS=':' read -r alpha2 max_step2 <<< "${EXPERIMENTS[$((i+1))]}"
        echo "[Experiment $((i+2))/$TOTAL]"
        run_experiment $alpha2 $max_step2
        PID2=$!
    else
        PID2=""
    fi
    
    echo ""
    echo "Batch $BATCH_NUM started with PIDs: $PID1 $PID2"
    
    # If not the last batch, wait 40 minutes
    if [ $((i+2)) -lt $TOTAL ]; then
        echo "Waiting 40 minutes before next batch..."
        echo "Next batch will start at approximately: $(date -d '+40 minutes')"
        echo ""
        echo "You can monitor current experiments with:"
        echo "  tail -f logs/stepwise_alpha${alpha1}_maxstep${max_step1}_*.log"
        if [ -n "$PID2" ]; then
            echo "  tail -f logs/stepwise_alpha${alpha2}_maxstep${max_step2}_*.log"
        fi
        echo ""
        sleep 2400  # 40 minutes = 2400 seconds
    fi
done

echo ""
echo "=========================================="
echo "ALL BATCHES SUBMITTED!"
echo "=========================================="
echo "Started at: $(date)"
echo ""
echo "To monitor progress:"
echo "  - Check experiment tracker: cat logs/experiment_tracker.txt"
echo "  - Monitor logs: tail -f logs/stepwise_*.log"
echo "  - Check running processes: ps aux | grep python3"
echo "  - Check GPU usage: watch nvidia-smi"
echo "  - List completed models: ls -la experiment/stepwise_*.pt"
echo ""
echo "To check detailed status:"
echo "  ./check_stepwise_status.sh"