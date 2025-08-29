#!/usr/bin/env python3
import sys
sys.path.append('/home/ubuntu/code/ucsf_rl')
from run_lstm_block_discrete_cql_with_logging import train_lstm_block_discrete_cql

print("Training LSTM Block Discrete CQL: alpha=0.0, vp2_bins=3")

try:
    agent, train_buffer, val_buffer, history = train_lstm_block_discrete_cql(
        alpha=0.0,
        vp2_bins=3,
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
