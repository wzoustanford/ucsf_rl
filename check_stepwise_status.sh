#!/bin/bash

echo "=========================================="
echo "STEPWISE CQL TRAINING STATUS"
echo "=========================================="
echo "Current time: $(date)"
echo ""

# Check running Python processes
echo "RUNNING EXPERIMENTS:"
echo "--------------------"
ps aux | grep "python3 run_unified_stepwise_cql" | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    # Try to extract alpha and max_step from command line
    if echo "$line" | grep -q "alpha"; then
        ALPHA=$(echo "$line" | grep -oP 'alpha \K[0-9.]+')
        MAXSTEP=$(echo "$line" | grep -oP 'max_step \K[0-9.]+')
        echo "  PID $PID: alpha=$ALPHA, max_step=$MAXSTEP"
    else
        echo "  PID $PID: (parameters unknown)"
    fi
done

if ! ps aux | grep "python3 run_unified_stepwise_cql" | grep -v grep > /dev/null; then
    echo "  No experiments currently running"
fi

echo ""
echo "COMPLETED MODELS:"
echo "-----------------"
# Group by alpha and max_step
for alpha in 0.0 0.0001 0.001 0.01; do
    for max_step in 0.1 0.2; do
        alpha_str=$(printf "%.6f" $alpha)
        max_step_str=$(printf "%.1f" $max_step)
        best_model="experiment/stepwise_cql_alpha${alpha_str}_maxstep${max_step_str}_best.pt"
        final_model="experiment/stepwise_cql_alpha${alpha_str}_maxstep${max_step_str}_final.pt"
        
        if [ -f "$best_model" ] || [ -f "$final_model" ]; then
            echo "  alpha=$alpha, max_step=$max_step:"
            if [ -f "$best_model" ]; then
                size=$(ls -lh "$best_model" | awk '{print $5}')
                modified=$(ls -l "$best_model" | awk '{print $6, $7, $8}')
                echo "    ✓ Best model: $size (saved: $modified)"
            fi
            if [ -f "$final_model" ]; then
                size=$(ls -lh "$final_model" | awk '{print $5}')
                modified=$(ls -l "$final_model" | awk '{print $6, $7, $8}')
                echo "    ✓ Final model: $size (saved: $modified)"
            fi
        fi
    done
done

echo ""
echo "RECENT LOG ACTIVITY:"
echo "--------------------"
# Show last 5 modified log files
ls -lt logs/stepwise_*.log 2>/dev/null | head -5 | while read line; do
    if [ ! -z "$line" ]; then
        filename=$(echo $line | awk '{print $NF}')
        modified=$(echo $line | awk '{print $6, $7, $8}')
        # Get last line from log
        if [ -f "$filename" ]; then
            last_line=$(tail -1 "$filename" 2>/dev/null | head -c 80)
            echo "  $filename"
            echo "    Modified: $modified"
            echo "    Last line: $last_line..."
        fi
    fi
done

echo ""
echo "GPU STATUS:"
echo "-----------"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r gpu_id gpu_name gpu_util mem_util mem_used mem_total; do
    echo "  GPU $gpu_id ($gpu_name): GPU=$gpu_util%, Mem=$mem_used/$mem_total MB ($mem_util%)"
done

echo ""
echo "DISK USAGE:"
echo "-----------"
echo -n "  experiment/ directory: "
du -sh experiment/ 2>/dev/null | awk '{print $1}'
echo -n "  logs/ directory: "
du -sh logs/ 2>/dev/null | awk '{print $1}'

echo ""
echo "=========================================="