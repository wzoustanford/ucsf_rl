#!/bin/bash
# Monitor Table 1 generation progress

echo "========================================"
echo " TABLE 1 GENERATION MONITOR"
echo "========================================"

# Check if process is running
PID=$(ps aux | grep generate_table1_block_discrete | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✓ Process running (PID: $PID)"
else
    echo "✗ Process not running"
fi

# Check log file
echo ""
echo "Progress:"
grep -c "Binary CQL..." logs/table1_block_discrete_evaluation.log | xargs echo "  Binary models started:"
grep -c "Dual Mixed CQL..." logs/table1_block_discrete_evaluation.log | xargs echo "  Dual Mixed models started:"
grep -c "Block Discrete CQL" logs/table1_block_discrete_evaluation.log | xargs echo "  Block Discrete models started:"

echo ""
echo "Latest activity:"
tail -3 logs/table1_block_discrete_evaluation.log

echo ""
echo "To watch live: tail -f logs/table1_block_discrete_evaluation.log"