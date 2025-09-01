#!/bin/bash
# Training script for Stepwise CQL with multiple alpha values
# Runs training with proper logging and monitoring

# Create logs directory
mkdir -p logs
mkdir -p experiment

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "======================================================================" 
echo " STARTING STEPWISE CQL TRAINING - MULTIPLE ALPHAS"
echo " Timestamp: $TIMESTAMP"
echo " Alphas: 0.00, 0.0001, 0.00001"
echo " Epochs: 200 per alpha"
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
" > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] ✅ Alpha=$alpha completed successfully"
    else
        echo "[$(date)] ❌ Alpha=$alpha failed - check $log_file for details"
    fi
    
    # Small delay between runs
    sleep 5
}

# Train all alphas sequentially
for alpha in 0.00 0.0001 0.00001; do
    train_alpha $alpha
done

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
echo "To monitor logs in real-time, use:"
echo "  tail -f logs/stepwise_cql_alpha<alpha>_${TIMESTAMP}.log"