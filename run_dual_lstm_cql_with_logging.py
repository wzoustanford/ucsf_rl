#!/usr/bin/env python3
"""
Dual Continuous LSTM-based CQL Training Script with Detailed Logging
=====================================================================
Combines LSTM architecture with dual continuous action space (VP1 and VP2).
Models both VP1 and VP2 as continuous variables with LSTM Q-functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

# Import our components
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from medical_sequence_buffer import MedicalSequenceBuffer, SequenceDataLoader

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


class Logger:
    """Simple logger that writes to both file and stdout."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', buffering=1)  # Line buffered
        
    def log(self, message: str):
        """Write message to both file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message, flush=True)
        self.log_file.write(full_message + '\n')
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


class LSTMDualQNetwork(nn.Module):
    """
    LSTM Q-Network for dual continuous actions.
    Processes sequences and outputs Q(s,a) for continuous action pairs.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,  # VP1 and VP2
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM Q-Network for dual continuous actions.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (2 for VP1 and VP2)
            hidden_dim: Hidden dimension for feature extraction
            lstm_hidden: LSTM hidden state dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        
        # Feature extraction layers (state + action concatenated)
        input_dim = state_dim + action_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer - single Q-value for the state-action pair
        self.q_head = nn.Linear(lstm_hidden, 1)
        
        # Better initialization
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.q_head.weight)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM Q-network.
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            actions: [batch_size, sequence_length, action_dim]
            hidden_state: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: [batch_size, sequence_length, 1] - Q(s,a) values
            new_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len, _ = states.shape
        
        # Ensure actions have correct dimension
        if actions.dim() == 2 and seq_len == 1:
            actions = actions.unsqueeze(1)
        
        # Concatenate states and actions
        state_action = torch.cat([states, actions], dim=-1)
        
        # Reshape for feature extraction
        state_action_flat = state_action.reshape(-1, self.state_dim + self.action_dim)
        features_flat = self.feature_extractor(state_action_flat)
        features = features_flat.reshape(batch_size, seq_len, -1)
        
        # Process through LSTM
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, states.device)
        
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Compute Q-values
        q_values = self.q_head(lstm_out)  # [batch_size, seq_len, 1]
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden).to(device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden).to(device)
        return (h0, c0)


class LSTMDualCQL:
    """
    Dual Continuous CQL with LSTM Q-networks for sequential learning.
    Handles continuous action space for both VP1 (0-1) and VP2 (0-0.5).
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.005,
        lr: float = 3e-4,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize LSTM Dual Continuous CQL."""
        self.state_dim = state_dim
        self.action_dim = 2  # VP1 and VP2
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        
        # Create LSTM Q-networks
        self.q1 = LSTMDualQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        self.q2 = LSTMDualQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        # Target networks
        self.q1_target = LSTMDualQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        self.q2_target = LSTMDualQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
    
    def select_actions_batch(
        self,
        states: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_samples: int = 10  # Reduced default for faster computation
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select best actions for a batch of states using sampling.
        SIMPLIFIED VERSION: Just sample random actions from the action space.
        This is much faster and often sufficient for CQL target computation.
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            hidden_state: Optional LSTM hidden state
            num_samples: Number of action samples to evaluate
        
        Returns:
            best_actions: [batch_size, sequence_length, 2]
            hidden_state: Updated hidden state
        """
        batch_size, seq_len, _ = states.shape
        
        with torch.no_grad():
            # For CQL, we can use a simpler strategy: sample random actions
            # This is much faster and often works well in practice
            vp1_actions = torch.rand(batch_size, seq_len, 1).to(self.device)
            vp2_actions = torch.rand(batch_size, seq_len, 1).to(self.device) * 0.5
            best_actions = torch.cat([vp1_actions, vp2_actions], dim=-1)
            
            # If we need to be more sophisticated, evaluate a small number of samples
            if num_samples > 1:
                # Sample multiple actions and pick best based on Q-values
                # But do it more efficiently in batch
                action_candidates = []
                for _ in range(num_samples):
                    vp1 = torch.rand(batch_size, seq_len, 1).to(self.device)
                    vp2 = torch.rand(batch_size, seq_len, 1).to(self.device) * 0.5
                    action_candidates.append(torch.cat([vp1, vp2], dim=-1))
                
                # Stack candidates
                action_candidates = torch.stack(action_candidates, dim=0)  # [samples, batch, seq, 2]
                
                # Evaluate first candidate to get hidden state
                q1_vals, hidden1 = self.q1(states, action_candidates[0], hidden_state)
                q2_vals, hidden2 = self.q2(states, action_candidates[0], hidden_state)
                best_q = torch.min(q1_vals, q2_vals).squeeze(-1)  # [batch, seq]
                best_actions = action_candidates[0]
                
                # Check remaining candidates
                for i in range(1, num_samples):
                    q1_vals, _ = self.q1(states, action_candidates[i], hidden_state)
                    q2_vals, _ = self.q2(states, action_candidates[i], hidden_state)
                    q_vals = torch.min(q1_vals, q2_vals).squeeze(-1)
                    
                    # Update best actions where new q_vals are better
                    better_mask = q_vals > best_q
                    best_actions[better_mask] = action_candidates[i][better_mask]
                    best_q = torch.maximum(best_q, q_vals)
                
                hidden_state = hidden1  # Use hidden from first evaluation
        
        return best_actions, hidden_state
    
    def compute_cql_loss_sequences(
        self,
        q_network: LSTMDualQNetwork,
        q_target_1: LSTMDualQNetwork,
        q_target_2: LSTMDualQNetwork,
        burn_in_states: torch.Tensor,
        burn_in_actions: torch.Tensor,
        training_states: torch.Tensor,
        training_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CQL loss for sequences with dual continuous actions.
        """
        batch_size = burn_in_states.shape[0]
        training_len = training_states.shape[1]
        
        # Burn-in phase: warm up LSTM hidden state (no gradients)
        hidden = None
        if burn_in_states.shape[1] > 0:
            with torch.no_grad():
                hidden = q_network.init_hidden(batch_size, self.device)
                _, hidden = q_network(burn_in_states, burn_in_actions, hidden)
        
        # Forward pass through training sequence
        q_values, _ = q_network(training_states, training_actions, hidden)
        q_values = q_values.squeeze(-1)  # [batch_size, training_len]
        
        # Compute target Q-values
        with torch.no_grad():
            # Select best next actions using sampling (reduced samples for speed)
            next_actions, _ = self.select_actions_batch(next_states, hidden, num_samples=5)
            
            # Get target Q-values
            next_q1, _ = q_target_1(next_states, next_actions, hidden)
            next_q2, _ = q_target_2(next_states, next_actions, hidden)
            next_q = torch.min(next_q1.squeeze(-1), next_q2.squeeze(-1))
            
            # Compute targets
            targets = rewards + self.gamma * next_q * (1 - dones)
            targets = torch.clamp(targets, -50, 50)  # Clip for stability
        
        # TD loss with importance sampling weights
        td_errors = q_values - targets
        td_loss = (td_errors ** 2).mean(dim=1)  # Average over sequence
        td_loss = (td_loss * weights).mean()
        
        # CQL regularization for continuous actions
        if self.alpha > 0:
            # Sample random actions for CQL penalty
            num_random = 5  # Reduced for speed
            random_vp1 = torch.rand(batch_size, training_len, num_random, 1).to(self.device)
            random_vp2 = torch.rand(batch_size, training_len, num_random, 1).to(self.device) * 0.5
            
            # Compute Q-values for random actions
            random_q_list = []
            for i in range(num_random):
                random_actions = torch.cat([random_vp1[:, :, i], random_vp2[:, :, i]], dim=-1)
                random_q, _ = q_network(training_states, random_actions, hidden)
                random_q_list.append(random_q.squeeze(-1))
            
            random_q_values = torch.stack(random_q_list, dim=-1)  # [batch, seq_len, num_random]
            
            # Conservative penalty with logsumexp for stability
            logsumexp = torch.logsumexp(random_q_values / 10.0, dim=-1) * 10.0
            cql_loss = (logsumexp - q_values).mean()
        else:
            cql_loss = torch.tensor(0.0).to(self.device)
        
        # Total loss
        total_loss = td_loss + self.alpha * cql_loss
        
        metrics = {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'q_values': q_values.mean().item(),
            'targets': targets.mean().item(),
            'td_errors': td_errors.abs().mean(dim=1)  # For priority updates
        }
        
        return total_loss, metrics
    
    def update_sequences(
        self,
        burn_in_batch: Dict[str, torch.Tensor],
        training_batch: Dict[str, torch.Tensor],
        weights: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks using sequence batch."""
        
        # Update Q1
        q1_loss, q1_metrics = self.compute_cql_loss_sequences(
            self.q1, self.q1_target, self.q2_target,
            burn_in_batch['states'],
            burn_in_batch['actions'],
            training_batch['states'],
            training_batch['actions'],
            training_batch['rewards'],
            training_batch['next_states'],
            training_batch['dones'],
            weights
        )
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        grad_norm1 = torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        q2_loss, q2_metrics = self.compute_cql_loss_sequences(
            self.q2, self.q1_target, self.q2_target,
            burn_in_batch['states'],
            burn_in_batch['actions'],
            training_batch['states'],
            training_batch['actions'],
            training_batch['rewards'],
            training_batch['next_states'],
            training_batch['dones'],
            weights
        )
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        grad_norm2 = torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Soft update target networks
        self.soft_update_targets()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_td_loss': q1_metrics['td_loss'],
            'q2_td_loss': q2_metrics['td_loss'],
            'q1_cql_loss': q1_metrics['cql_loss'],
            'q2_cql_loss': q2_metrics['cql_loss'],
            'q_values': q1_metrics['q_values'],
            'q1_values': q1_metrics['q_values'],
            'q2_values': q2_metrics['q_values'],
            'grad_norm': (grad_norm1 + grad_norm2) / 2,
            'td_errors': q1_metrics.get('td_errors')
        }
    
    def soft_update_targets(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def evaluate_lstm_dual_cql(
    agent: LSTMDualCQL,
    sequence_buffer: MedicalSequenceBuffer,
    num_eval_batches: int = 10,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate LSTM Dual CQL on validation sequences.
    """
    agent.q1.eval()
    agent.q2.eval()
    
    metrics = {
        'q_values': [],
        'q1_values': [],
        'q2_values': [],
        'q_variance': []
    }
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            # Sample sequences
            burn_in_batch, training_batch, _, weights = sequence_buffer.sample_sequences(batch_size)
            
            # Convert to torch
            burn_in_torch, training_torch, weights_torch = SequenceDataLoader.prepare_torch_batch(
                burn_in_batch, training_batch, weights, device=agent.device
            )
            
            # Get Q-values
            hidden1 = agent.q1.init_hidden(batch_size, agent.device)
            hidden2 = agent.q2.init_hidden(batch_size, agent.device)
            
            # Burn-in
            if burn_in_torch['states'].shape[1] > 0:
                _, hidden1 = agent.q1(burn_in_torch['states'], burn_in_torch['actions'], hidden1)
                _, hidden2 = agent.q2(burn_in_torch['states'], burn_in_torch['actions'], hidden2)
            
            # Get Q-values for training sequence
            q1_vals, _ = agent.q1(training_torch['states'], training_torch['actions'], hidden1)
            q2_vals, _ = agent.q2(training_torch['states'], training_torch['actions'], hidden2)
            
            q_vals = torch.min(q1_vals, q2_vals)
            
            metrics['q_values'].append(q_vals.mean().item())
            metrics['q1_values'].append(q1_vals.mean().item())
            metrics['q2_values'].append(q2_vals.mean().item())
            metrics['q_variance'].append(q_vals.var().item())
    
    return {
        'mean_q_value': np.mean(metrics['q_values']),
        'std_q_value': np.std(metrics['q_values']),
        'mean_q1_value': np.mean(metrics['q1_values']),
        'mean_q2_value': np.mean(metrics['q2_values']),
        'mean_q_variance': np.mean(metrics['q_variance'])
    }


def train_lstm_dual_cql(
    alpha: float = 0.001,
    sequence_length: int = 20,
    burn_in_length: int = 8,
    overlap: int = 10,
    hidden_dim: int = 64,
    lstm_hidden: int = 64,
    num_lstm_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    tau: float = 0.8,
    gamma: float = 0.95,
    grad_clip: float = 1.0,
    buffer_capacity: int = 50000,
    save_dir: str = 'experiment',
    log_dir: str = 'logs',
    log_every: int = 1
):
    """
    Train LSTM-based Dual Continuous CQL with detailed logging.
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f'lstm_dual_cql_alpha{alpha:.4f}_{timestamp}.log')
    logger = Logger(log_path)
    
    logger.log("="*70)
    logger.log(f" LSTM DUAL CONTINUOUS CQL TRAINING WITH ALPHA={alpha}")
    logger.log("="*70)
    
    # Save configuration to JSON
    config = {
        'model_type': 'dual_continuous_lstm',
        'alpha': alpha,
        'sequence_length': sequence_length,
        'burn_in_length': burn_in_length,
        'overlap': overlap,
        'hidden_dim': hidden_dim,
        'lstm_hidden': lstm_hidden,
        'num_lstm_layers': num_lstm_layers,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'tau': tau,
        'gamma': gamma,
        'grad_clip': grad_clip,
        'buffer_capacity': buffer_capacity,
        'timestamp': timestamp
    }
    
    config_path = os.path.join(log_dir, f'lstm_dual_cql_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.log(f"Configuration saved to {config_path}")
    
    # Initialize data pipeline for dual continuous actions
    logger.log("\nInitializing data pipeline...")
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    state_dim = train_data['states'].shape[1]
    
    # Print configuration
    logger.log("\n" + "="*70)
    logger.log("CONFIGURATION:")
    logger.log(f"  State dimension: {state_dim}")
    logger.log(f"  Action dimension: 2 (VP1: 0-1, VP2: 0-0.5)")
    logger.log(f"  Sequence length: {sequence_length}")
    logger.log(f"  Burn-in length: {burn_in_length}")
    logger.log(f"  Training length: {sequence_length - burn_in_length}")
    logger.log(f"  Overlap: {overlap}")
    logger.log(f"  Hidden dim: {hidden_dim}")
    logger.log(f"  LSTM hidden: {lstm_hidden}")
    logger.log(f"  LSTM layers: {num_lstm_layers}")
    logger.log(f"  Alpha (CQL penalty): {alpha}")
    logger.log(f"  Tau (soft update): {tau}")
    logger.log(f"  Learning rate: {learning_rate}")
    logger.log(f"  Batch size: {batch_size}")
    logger.log(f"  Epochs: {epochs}")
    logger.log(f"  Gradient clipping: {grad_clip}")
    logger.log("="*70)
    
    # Create sequence buffers
    logger.log("\nCreating sequence buffers...")
    
    train_buffer = MedicalSequenceBuffer(
        capacity=buffer_capacity,
        sequence_length=sequence_length,
        burn_in_length=burn_in_length,
        overlap=overlap,
        priority_type='mortality_weighted'
    )
    
    val_buffer = MedicalSequenceBuffer(
        capacity=buffer_capacity // 5,
        sequence_length=sequence_length,
        burn_in_length=burn_in_length,
        overlap=overlap,
        priority_type='uniform'
    )
    
    # Fill training buffer
    logger.log("Generating training sequences...")
    for patient_id, (start_idx, end_idx) in pipeline.train_patient_groups.items():
        for t in range(start_idx, end_idx):
            train_buffer.add_transition(
                state=train_data['states'][t],
                action=train_data['actions'][t],  # Now 2D: [VP1, VP2]
                reward=train_data['rewards'][t],
                next_state=train_data['next_states'][t],
                done=bool(train_data['dones'][t]),
                patient_id=patient_id
            )
    
    logger.log(f"  Generated {len(train_buffer)} training sequences from {len(pipeline.train_patient_groups)} patients")
    
    # Fill validation buffer
    logger.log("Generating validation sequences...")
    for patient_id, (start_idx, end_idx) in pipeline.val_patient_groups.items():
        for t in range(start_idx, end_idx):
            val_buffer.add_transition(
                state=val_data['states'][t],
                action=val_data['actions'][t],  # Now 2D: [VP1, VP2]
                reward=val_data['rewards'][t],
                next_state=val_data['next_states'][t],
                done=bool(val_data['dones'][t]),
                patient_id=patient_id
            )
    
    logger.log(f"  Generated {len(val_buffer)} validation sequences from {len(pipeline.val_patient_groups)} patients")
    
    # Initialize LSTM Dual CQL agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"\nInitializing LSTM Dual CQL agent on {device}...")
    
    agent = LSTMDualCQL(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_lstm_layers,
        alpha=alpha,
        gamma=gamma,
        tau=tau,
        lr=learning_rate,
        grad_clip=grad_clip,
        device=device
    )
    
    total_params = sum(p.numel() for p in agent.q1.parameters())
    logger.log(f"  Total parameters per Q-network: {total_params:,}")
    
    # Training loop
    logger.log(f"\nStarting training for {epochs} epochs...")
    logger.log("="*70)
    start_time = time.time()
    
    best_val_q = -float('inf')
    batches_per_epoch = min(len(train_buffer) // batch_size, 500)  # Reduced from 500 for faster training
    
    logger.log(f"Batches per epoch: {batches_per_epoch}")
    logger.log("")
    
    # CSV header for easy parsing
    logger.log("Epoch,Time,Q1_Loss,Q2_Loss,TD_Loss,CQL_Loss,Train_Q,Train_Q1,Train_Q2,Val_Q,Val_Q1,Val_Q2,Val_Q_Std")
    
    training_history = {
        'train_loss': [],
        'train_q_values': [],
        'val_q_values': [],
        'cql_penalties': [],
        'td_losses': []
    }
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        agent.q1.train()
        agent.q2.train()
        
        epoch_metrics = {
            'q1_loss': 0,
            'q2_loss': 0,
            'q1_td_loss': 0,
            'q2_td_loss': 0,
            'q1_cql_loss': 0,
            'q2_cql_loss': 0,
            'q_values': 0,
            'q1_values': 0,
            'q2_values': 0,
            'grad_norm': 0
        }
        
        for batch_idx in range(batches_per_epoch):
            if batch_idx == 0:
                logger.log(f"  Starting batch {batch_idx+1}/{batches_per_epoch}...")
            
            # Sample sequences
            burn_in_batch, training_batch, indices, weights = train_buffer.sample_sequences(batch_size)
            
            # Convert to torch tensors
            burn_in_torch, training_torch, weights_torch = SequenceDataLoader.prepare_torch_batch(
                burn_in_batch, training_batch, weights, device=device
            )
            
            # Update agent
            batch_start = time.time()
            metrics = agent.update_sequences(burn_in_torch, training_torch, weights_torch)
            
            if batch_idx == 0:
                batch_time = time.time() - batch_start
                logger.log(f"  First batch took {batch_time:.2f} seconds")
            
            # Update priorities based on TD error
            if hasattr(train_buffer, 'update_priorities'):
                td_errors = metrics.get('td_errors', None)
                if td_errors is not None:
                    train_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
            
            # Accumulate metrics
            for key in metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= batches_per_epoch
        
        # Store training metrics
        training_history['train_loss'].append(epoch_metrics['q1_loss'])
        training_history['train_q_values'].append(epoch_metrics['q_values'])
        training_history['td_losses'].append(epoch_metrics['q1_td_loss'])
        training_history['cql_penalties'].append(epoch_metrics['q1_cql_loss'])
        
        # Validation phase
        val_metrics = evaluate_lstm_dual_cql(agent, val_buffer, num_eval_batches=10, batch_size=batch_size)
        training_history['val_q_values'].append(val_metrics['mean_q_value'])
        
        # Time for epoch
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Log metrics in CSV format
        logger.log(f"{epoch+1},{total_time/60:.2f},"
                  f"{epoch_metrics['q1_loss']:.6f},{epoch_metrics['q2_loss']:.6f},"
                  f"{epoch_metrics['q1_td_loss']:.6f},{epoch_metrics['q1_cql_loss']:.6f},"
                  f"{epoch_metrics['q_values']:.6f},{epoch_metrics['q1_values']:.6f},{epoch_metrics['q2_values']:.6f},"
                  f"{val_metrics['mean_q_value']:.6f},{val_metrics['mean_q1_value']:.6f},"
                  f"{val_metrics['mean_q2_value']:.6f},{val_metrics['std_q_value']:.6f}")
        
        # Save best model
        if val_metrics['mean_q_value'] > best_val_q:
            best_val_q = val_metrics['mean_q_value']
            
            save_path = os.path.join(save_dir, f'lstm_dual_cql_alpha{alpha:.2f}_best.pt')
            torch.save({
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'q1_target_state_dict': agent.q1_target.state_dict(),
                'q2_target_state_dict': agent.q2_target.state_dict(),
                'config': config,
                'epoch': epoch + 1,
                'best_val_q': best_val_q,
                'training_history': training_history
            }, save_path)
            
            if (epoch + 1) % 10 == 0:
                logger.log(f"  >>> New best model saved at epoch {epoch+1} with Val Q={best_val_q:.6f}")
        
        # Detailed logging every N epochs
        if (epoch + 1) % 5 == 0:
            logger.log("")
            logger.log(f"Epoch {epoch+1}/{epochs} Summary:")
            logger.log(f"  Total Time: {total_time/60:.1f} min")
            logger.log(f"  Q1 Loss: {epoch_metrics['q1_loss']:.6f}")
            logger.log(f"  Q2 Loss: {epoch_metrics['q2_loss']:.6f}")
            logger.log(f"  TD Loss (Q1): {epoch_metrics['q1_td_loss']:.6f}")
            logger.log(f"  CQL Penalty (Q1): {epoch_metrics['q1_cql_loss']:.6f}")
            logger.log(f"  Train Q-value: {epoch_metrics['q_values']:.6f}")
            logger.log(f"  Val Q-value: {val_metrics['mean_q_value']:.6f} ± {val_metrics['std_q_value']:.6f}")
            logger.log(f"  Best Val Q: {best_val_q:.6f}")
            logger.log("")
    
    # Save final model
    final_save_path = os.path.join(save_dir, f'lstm_dual_cql_alpha{alpha:.2f}_final.pt')
    torch.save({
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'q1_target_state_dict': agent.q1_target.state_dict(),
        'q2_target_state_dict': agent.q2_target.state_dict(),
        'config': config,
        'epoch': epochs,
        'training_history': training_history
    }, final_save_path)
    
    total_time = time.time() - start_time
    logger.log("\n" + "="*70)
    logger.log(f"✅ LSTM Dual CQL training completed in {total_time/60:.1f} minutes!")
    logger.log("Models saved:")
    logger.log(f"  - {save_path} (best)")
    logger.log(f"  - {final_save_path} (final)")
    logger.log(f"  - Log file: {log_path}")
    
    # Final statistics
    logger.log("\nFinal Statistics:")
    logger.log(f"  Best validation Q-value: {best_val_q:.6f}")
    logger.log(f"  Final train Q-value: {epoch_metrics['q_values']:.6f}")
    logger.log(f"  Final validation Q-value: {val_metrics['mean_q_value']:.6f}")
    logger.log("="*70)
    
    logger.close()
    
    return agent, train_buffer, val_buffer, training_history


def main():
    """Main training function."""
    print("="*70)
    print(" LSTM DUAL CONTINUOUS CQL TRAINING WITH LOGGING")
    print("="*70)
    print("\nTraining LSTM-based Dual Continuous CQL with detailed logging")
    print("Log files will be saved to logs/ directory")
    print("Models will be saved to experiment/ directory")
    
    # Train with different alpha values
    alphas = [0.001]  # Can add more values like [0.0, 0.001, 0.01] for comparison
    
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f" Training with alpha={alpha}")
        print(f"{'='*70}")
        
        agent, train_buffer, val_buffer, history = train_lstm_dual_cql(
            alpha=alpha,
            sequence_length=20,
            burn_in_length=8,
            overlap=10,
            hidden_dim=64,
            lstm_hidden=64,
            num_lstm_layers=2,
            batch_size=32,
            epochs=100,  # Full training
            learning_rate=1e-3,
            tau=0.8,
            gamma=0.95,
            grad_clip=1.0,
            buffer_capacity=50000,
            log_every=1
        )
    
    print("\n" + "="*70)
    print(" ALL TRAINING COMPLETE")
    print("="*70)
    print("\nCheck logs/ directory for detailed training logs")
    print("Check experiment/ directory for saved models")


if __name__ == "__main__":
    main()