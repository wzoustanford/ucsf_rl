#!/bin/bash
# Script to monitor LSTM training progress

echo "======================================================================" 
echo " LSTM BLOCK DISCRETE CQL TRAINING MONITOR"
echo " Current time: $(date)"
echo "======================================================================"
echo ""

# Check for running Python LSTM training processes
echo "Active LSTM training processes:"
ps aux | grep -E "python3.*train_lstm_block_discrete_cql" | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    START_TIME=$(echo $line | awk '{print $9}')
    # Try to extract alpha and bins from command
    if echo "$line" | grep -q "alpha="; then
        ALPHA=$(echo "$line" | sed -n 's/.*alpha=\([0-9.]*\).*/\1/p')
        BINS=$(echo "$line" | sed -n 's/.*vp2_bins=\([0-9]*\).*/\1/p')
        echo "  PID $PID (started at $START_TIME) - alpha=$ALPHA, bins=$BINS"
    else
        echo "  PID $PID (started at $START_TIME)"
    fi
done

if ! ps aux | grep -E "python3.*train_lstm_block_discrete_cql" | grep -v grep > /dev/null; then
    echo "  No active LSTM training processes found"
fi

echo ""
echo "Latest LSTM log files:"
ls -lt logs/lstm_cql_*.log 2>/dev/null | head -6 | while read line; do
    echo "  $line"
done

echo ""
echo "Training progress from latest logs:"
for log in $(ls -t logs/lstm_cql_*.log 2>/dev/null | head -6); do
    # Extract alpha and bins from filename
    filename=$(basename $log)
    alpha=$(echo $filename | sed 's/lstm_cql_alpha\([^_]*\)_.*/\1/')
    bins=$(echo $filename | sed 's/.*_bins\([0-9]*\)_.*/\1/')
    
    echo ""
    echo "Alpha=$alpha, bins=$bins ($filename):"
    
    # Check if training is complete
    if grep -q "✅ Training completed successfully" "$log" 2>/dev/null; then
        echo "  Status: ✅ COMPLETED"
        # Show final metrics
        tail -30 "$log" | grep -E "(Epoch 100:|completed in|Best validation)" | tail -3 | sed 's/^/  /'
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
        echo "  Epochs completed: $EPOCH_COUNT / 100"
    fi
done

echo ""
echo "Saved LSTM models in experiment/:"
ls -lh experiment/lstm_block_discrete_cql_*.pt 2>/dev/null | tail -12 | while read line; do
    echo "  $line"
done

echo ""
echo "Completed configurations:"
for alpha in 0.0000 0.0001; do
    for bins in 3 5 10; do
        if ls experiment/lstm_block_discrete_cql_alpha${alpha}_bins${bins}_final.pt >/dev/null 2>&1; then
            echo "  ✅ Alpha=$alpha, bins=$bins"
        fi
    done
done

echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
    echo "  $line"
done

echo ""
echo "To see detailed progress for a specific configuration, use:"
echo "  tail -f logs/lstm_cql_alpha<alpha>_bins<bins>_*.log"
echo ""
echo "To continuously monitor this summary, use:"
echo "  watch -n 30 ./monitor_lstm_training.sh"