#!/bin/bash

# LSTM Training Status Monitor

echo "=========================================="
echo " LSTM TRAINING STATUS"
echo " $(date)"
echo "=========================================="

# Check running processes
echo ""
echo "RUNNING PROCESSES:"
echo "------------------"
ps aux | grep "run_lstm_block_discrete_cql" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    
    # Extract parameters from command
    if [[ $cmd == *"--alpha"* ]]; then
        alpha=$(echo $cmd | sed -n 's/.*--alpha \([0-9.]*\).*/\1/p')
        bins=$(echo $cmd | sed -n 's/.*--vp2_bins \([0-9]*\).*/\1/p')
        echo "  PID $pid: alpha=$alpha, bins=$bins"
        
        # Show latest log line if log file exists
        log_file=$(ls -t logs/lstm_alpha${alpha}_bins${bins}_*.log 2>/dev/null | head -1)
        if [ -f "$log_file" ]; then
            latest=$(tail -1 "$log_file" | head -c 100)
            echo "    Latest: $latest"
        fi
    fi
done

if ! ps aux | grep "run_lstm_block_discrete_cql" | grep -v grep > /dev/null; then
    echo "  No LSTM training processes currently running"
fi

# Check completed models
echo ""
echo "COMPLETED MODELS:"
echo "-----------------"
model_count=0
for model in experiment/lstm_block_discrete_cql_alpha*_bins*.pt; do
    if [ -f "$model" ]; then
        basename "$model"
        ((model_count++))
    fi
done

if [ $model_count -eq 0 ]; then
    echo "  No completed models yet"
fi

# Check log files
echo ""
echo "LOG FILES (most recent first):"
echo "-------------------------------"
ls -lt logs/lstm_alpha*.log 2>/dev/null | head -5 | while read line; do
    file=$(echo $line | awk '{print $9}')
    size=$(echo $line | awk '{print $5}')
    time=$(echo $line | awk '{print $6" "$7" "$8}')
    
    if [ -f "$file" ]; then
        # Extract parameters from filename
        alpha=$(echo $file | sed -n 's/.*alpha\([0-9.]*\)_.*/\1/p')
        bins=$(echo $file | sed -n 's/.*bins\([0-9]*\)_.*/\1/p')
        
        # Check if file is being written to
        if lsof "$file" > /dev/null 2>&1; then
            status="[ACTIVE]"
        else
            status="[COMPLETE]"
        fi
        
        echo "  $status alpha=$alpha, bins=$bins - $file (size: $size, modified: $time)"
    fi
done

# GPU status
echo ""
echo "GPU STATUS:"
echo "-----------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
    echo "  GPU $line"
done

# Summary
echo ""
echo "=========================================="
echo "SUMMARY:"
total_expected=2
completed=$model_count
running=$(ps aux | grep "run_lstm_block_discrete_cql" | grep -v grep | wc -l)
remaining=$((total_expected - completed - running))

echo "  Total experiments: $total_expected"
echo "  Completed: $completed"
echo "  Running: $running"
echo "  Remaining: $remaining"
echo "=========================================="