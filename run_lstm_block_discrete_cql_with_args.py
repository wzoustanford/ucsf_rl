#!/usr/bin/env python3
"""
LSTM Block Discrete CQL Training Script with Command-line Arguments
===================================================================
Trains LSTM CQL with discrete dosing levels for Norepinephrine.
"""

import numpy as np
import torch
import time
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple

# Import our components
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from medical_sequence_buffer import MedicalSequenceBuffer, SequenceDataLoader
from lstm_block_discrete_cql_network import LSTMBlockDiscreteCQL

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Import the original training functions
from run_lstm_block_discrete_cql_with_logging import (
    Logger, evaluate_lstm_cql, train_lstm_block_discrete_cql
)


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Train LSTM Block Discrete CQL')
    parser.add_argument('--alpha', type=float, required=True,
                        help='CQL regularization parameter')
    parser.add_argument('--vp2_bins', type=int, required=True,
                        help='Number of discrete bins for VP2')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    args = parser.parse_args()
    
    print("="*70, flush=True)
    print(" LSTM BLOCK DISCRETE CQL TRAINING", flush=True)
    print("="*70, flush=True)
    print(f"\nTraining configuration:", flush=True)
    print(f"  - Alpha: {args.alpha}", flush=True)
    print(f"  - VP2 bins: {args.vp2_bins}", flush=True)
    print(f"  - Epochs: {args.epochs}", flush=True)
    print(f"  - Sequence length: 20", flush=True)
    print(f"  - Hidden dim: 64", flush=True)
    print(f"  - LSTM layers: 2", flush=True)
    print(f"  - Batch size: 32", flush=True)
    print(f"  - Learning rate: 1e-3", flush=True)
    print(f"  - Tau: 0.8", flush=True)
    print("="*70, flush=True)
    
    # Train with specified parameters
    agent, train_buffer, val_buffer, history = train_lstm_block_discrete_cql(
        alpha=args.alpha,
        vp2_bins=args.vp2_bins,
        sequence_length=5,
        burn_in_length=2,
        overlap=1,
        hidden_dim=32,
        lstm_hidden=32,
        num_lstm_layers=2,
        batch_size=32,
        epochs=args.epochs,
        learning_rate=1e-3,
        tau=0.8,
        gamma=0.95,
        grad_clip=1.0,
        buffer_capacity=150000,
        log_every=1  # Log every epoch
    )
    
    print("\n" + "="*70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"\nModel saved: lstm_block_discrete_cql_alpha{args.alpha:.6f}_bins{args.vp2_bins}_*.pt", flush=True)
    print("Check logs/ directory for detailed training logs", flush=True)
    print("Check experiment/ directory for saved models", flush=True)


if __name__ == "__main__":
    main()