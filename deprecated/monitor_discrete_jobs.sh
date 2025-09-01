#!/bin/bash

# Simple monitor for discrete CQL jobs
# Usage: ./monitor_discrete_jobs.sh [log_dir]

if [ -z "$1" ]; then
    # Find most recent log directory
    LOG_DIR=$(ls -dt logs/discrete_cql_* 2>/dev/null | head -1)
    if [ -z "$LOG_DIR" ]; then
        echo "No log directory found. Please specify one."
        echo "Usage: $0 <log_dir>"
        exit 1
    fi
else
    LOG_DIR="$1"
fi

echo "Monitoring: $LOG_DIR"

PID_FILE="$LOG_DIR/running_pids.txt"
if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found: $PID_FILE"
    exit 1
fi

while true; do
    clear
    echo "========================================"
    echo "DISCRETE CQL JOB MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    TOTAL=0
    RUNNING=0
    COMPLETED=0
    FAILED=0
    
    # Check each job
    while IFS= read -r line; do
        PID=$(echo $line | awk '{print $1}')
        JOB_NAME=$(echo $line | awk '{print $2}')
        TOTAL=$((TOTAL + 1))
        
        if kill -0 $PID 2>/dev/null; then
            # Process is running
            RUNNING=$((RUNNING + 1))
            echo "🔄 $JOB_NAME (PID: $PID) - RUNNING"
            
            # Show last log line with epoch info
            LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
            if [ -f "$LOG_FILE" ]; then
                LAST_EPOCH=$(grep -E "Epoch [0-9]+" "$LOG_FILE" | tail -1)
                if [ ! -z "$LAST_EPOCH" ]; then
                    echo "   $LAST_EPOCH"
                fi
            fi
            
            # Check GPU memory usage for this process
            if command -v nvidia-smi &> /dev/null; then
                GPU_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | grep "^$PID" | awk -F', ' '{print $2}')
                if [ ! -z "$GPU_MEM" ]; then
                    echo "   GPU Memory: ${GPU_MEM} MB"
                fi
            fi
        else
            # Process is not running - check if completed or failed
            EXP_DIR="${LOG_DIR/logs/experiment}"
            RESULT_FILE="$EXP_DIR/${JOB_NAME}/result.json"
            ERROR_FILE="$EXP_DIR/${JOB_NAME}/error.json"
            
            if [ -f "$RESULT_FILE" ]; then
                COMPLETED=$((COMPLETED + 1))
                VAL_LOSS=$(python3 -c "import json; print(f\"{json.load(open('$RESULT_FILE'))['best_val_loss']:.4f}\")" 2>/dev/null || echo "N/A")
                echo "✅ $JOB_NAME - COMPLETED (Val Loss: $VAL_LOSS)"
            elif [ -f "$ERROR_FILE" ]; then
                FAILED=$((FAILED + 1))
                echo "❌ $JOB_NAME - FAILED"
                # Show error
                ERR_FILE="$LOG_DIR/${JOB_NAME}.err"
                if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
                    echo "   Error: $(tail -1 $ERR_FILE)"
                fi
            else
                FAILED=$((FAILED + 1))
                echo "❓ $JOB_NAME - STOPPED (Unknown status)"
            fi
        fi
    done < "$PID_FILE"
    
    echo ""
    echo "========================================"
    echo "SUMMARY: Total=$TOTAL, Running=$RUNNING, Completed=$COMPLETED, Failed=$FAILED"
    
    # Show overall GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F', ' '{printf "  GPU Util: %s%%, Memory: %s/%s MB\n", $1, $2, $3}'
    fi
    
    echo "========================================"
    
    if [ $RUNNING -eq 0 ]; then
        echo ""
        echo "All jobs finished!"
        break
    fi
    
    echo ""
    echo "Press Ctrl+C to exit (jobs will continue running)"
    echo "Refreshing in 30 seconds..."
    sleep 30
done

echo ""
echo "Final Results Summary:"
echo "====================="

# Show all results sorted by validation loss
EXP_DIR="${LOG_DIR/logs/experiment}"
for result_file in $EXP_DIR/*/result.json; do
    if [ -f "$result_file" ]; then
        python3 -c "
import json
import os
with open('$result_file') as f:
    r = json.load(f)
    dirname = os.path.dirname('$result_file').split('/')[-1]
    print(f\"{dirname}: Val Loss = {r['best_val_loss']:.4f}\")
" 2>/dev/null
    fi
done | sort -t'=' -k2 -n