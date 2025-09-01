#!/bin/bash
# Script to monitor training progress

echo "======================================================================" 
echo " STEPWISE CQL TRAINING MONITOR"
echo " Current time: $(date)"
echo "======================================================================"
echo ""

# Check for running Python training processes
echo "Active training processes:"
ps aux | grep -E "python3.*train_unified_stepwise_cql" | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    START_TIME=$(echo $line | awk '{print $9}')
    echo "  PID $PID (started at $START_TIME)"
done

if ! ps aux | grep -E "python3.*train_unified_stepwise_cql" | grep -v grep > /dev/null; then
    echo "  No active training processes found"
fi

echo ""
echo "Latest log files:"
ls -lt logs/stepwise_cql_*.log 2>/dev/null | head -5 | while read line; do
    echo "  $line"
done

echo ""
echo "Training progress from latest logs:"
for log in $(ls -t logs/stepwise_cql_*.log 2>/dev/null | head -4); do
    alpha=$(basename $log | sed 's/stepwise_cql_alpha\([^_]*\).*/\1/')
    echo ""
    echo "Alpha=$alpha ($(basename $log)):"
    
    # Check if training is complete
    if grep -q "✅ Training completed successfully" "$log" 2>/dev/null; then
        echo "  Status: ✅ COMPLETED"
        # Show final metrics
        tail -20 "$log" | grep -E "(Epoch 200:|completed in|Q1 Loss|Val Q-value)" | tail -3 | sed 's/^/  /'
    elif grep -q "❌ Training failed" "$log" 2>/dev/null; then
        echo "  Status: ❌ FAILED"
        # Show error
        grep -A5 "Training failed" "$log" | head -6 | sed 's/^/  /'
    else
        # Training in progress - show latest epoch
        echo "  Status: 🔄 IN PROGRESS"
        # Get latest epoch info
        grep "Epoch" "$log" 2>/dev/null | tail -1 | sed 's/^/  /'
        # Count total epochs so far
        EPOCH_COUNT=$(grep -c "Epoch" "$log" 2>/dev/null)
        echo "  Epochs completed: $EPOCH_COUNT / 200"
    fi
done

echo ""
echo "Saved models in experiment/:"
ls -lh experiment/stepwise_cql_*.pt 2>/dev/null | tail -8 | while read line; do
    echo "  $line"
done

echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
    echo "  $line"
done

echo ""
echo "To see detailed progress for a specific alpha, use:"
echo "  tail -f logs/stepwise_cql_alpha<alpha>_*.log"
echo ""
echo "To continuously monitor this summary, use:"
echo "  watch -n 30 ./monitor_training.sh"