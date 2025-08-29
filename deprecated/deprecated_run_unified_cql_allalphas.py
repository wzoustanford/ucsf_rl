#!/usr/bin/env python3
"""
Unified training script for Binary and Dual CQL with multiple alpha values.

Binary CQL: VP1 is binary (0 or 1)
Dual CQL: VP1 is binary (0 or 1), VP2 is continuous (0 to 0.5 mcg/kg/min)

Trains models with alpha values: 0.0, 0.001, 0.01
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os
import sys
from typing import Dict, Tuple, Optional

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Binary CQL Implementation
# ============================================================================

class BinaryQNetwork(nn.Module):
    """Q-network for binary actions Q(s,a) where a ∈ {0,1}"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Network takes state + action as input
        input_dim = state_dim + 1
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, 1] - binary action
        Returns:
            q_value: [batch_size, 1]
        """
        if action.dim() == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class BinaryCQL:
    """Binary CQL implementation for VP1 only"""
    
    def __init__(
        self,
        state_dim: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.005,
        lr: float = 3e-4,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks
        self.q1 = BinaryQNetwork(state_dim).to(device)
        self.q2 = BinaryQNetwork(state_dim).to(device)
        self.q1_target = BinaryQNetwork(state_dim).to(device)
        self.q2_target = BinaryQNetwork(state_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks"""
        
        # Ensure actions are properly shaped
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            # Evaluate both actions at next state
            next_q1_a0 = self.q1_target(next_states, torch.zeros_like(actions))
            next_q1_a1 = self.q1_target(next_states, torch.ones_like(actions))
            next_q2_a0 = self.q2_target(next_states, torch.zeros_like(actions))
            next_q2_a1 = self.q2_target(next_states, torch.ones_like(actions))
            
            # Take max over actions, then min over networks
            next_q1_max = torch.max(next_q1_a0, next_q1_a1).squeeze()
            next_q2_max = torch.max(next_q2_a0, next_q2_a1).squeeze()
            next_q = torch.min(next_q1_max, next_q2_max)
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Compute Q-values for both actions
            q1_a0 = self.q1(states, torch.zeros_like(actions)).squeeze()
            q1_a1 = self.q1(states, torch.ones_like(actions)).squeeze()
            
            # Log-sum-exp over actions
            q1_all = torch.stack([q1_a0, q1_a1], dim=1)
            cql1_loss = torch.logsumexp(q1_all, dim=1).mean() - current_q1.mean()
        else:
            cql1_loss = torch.tensor(0.0)
        
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # CQL penalty for Q2
        if self.alpha > 0:
            q2_a0 = self.q2(states, torch.zeros_like(actions)).squeeze()
            q2_a1 = self.q2(states, torch.ones_like(actions)).squeeze()
            q2_all = torch.stack([q2_a0, q2_a1], dim=1)
            cql2_loss = torch.logsumexp(q2_all, dim=1).mean() - current_q2.mean()
        else:
            cql2_loss = torch.tensor(0.0)
        
        total_q2_loss = q2_loss + self.alpha * cql2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if isinstance(cql1_loss, torch.Tensor) else cql1_loss,
            'cql2_loss': cql2_loss.item() if isinstance(cql2_loss, torch.Tensor) else cql2_loss,
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item()
        }
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau
        }, path)


# ============================================================================
# Dual CQL Implementation (VP1 Binary, VP2 Continuous)
# ============================================================================

class DualMixedQNetwork(nn.Module):
    """
    Q-network for mixed actions: VP1 binary, VP2 continuous
    Q(s, vp1, vp2) where vp1 ∈ {0,1} and vp2 ∈ [0, 0.5]
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Network takes state + vp1 (binary) + vp2 (continuous) as input
        input_dim = state_dim + 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, 2] - [vp1_binary, vp2_continuous]
        Returns:
            q_value: [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DualMixedCQL:
    """Dual CQL with VP1 binary and VP2 continuous"""
    
    def __init__(
        self,
        state_dim: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.005,
        lr: float = 3e-4,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks
        self.q1 = DualMixedQNetwork(state_dim).to(device)
        self.q2 = DualMixedQNetwork(state_dim).to(device)
        self.q1_target = DualMixedQNetwork(state_dim).to(device)
        self.q2_target = DualMixedQNetwork(state_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def select_action(self, state: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        """
        Select best action for a given state by sampling.
        VP1 is binary, VP2 is continuous [0, 0.5]
        """
        with torch.no_grad():
            batch_size = state.shape[0]
            
            # Sample actions: VP1 binary, VP2 continuous
            vp1_samples = torch.randint(0, 2, (batch_size, num_samples, 1)).float().to(self.device)
            vp2_samples = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            action_samples = torch.cat([vp1_samples, vp2_samples], dim=-1)
            
            # Evaluate Q-values for all samples (batched)
            # Reshape for batch processing: [batch_size * num_samples, action_dim]
            state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)  # [batch, samples, state_dim]
            state_flat = state_expanded.reshape(-1, state.shape[-1])  # [batch*samples, state_dim]
            action_flat = action_samples.reshape(-1, 2)  # [batch*samples, 2]
            
            # Single forward pass for all samples
            q1_vals = self.q1(state_flat, action_flat).reshape(batch_size, num_samples)
            q2_vals = self.q2(state_flat, action_flat).reshape(batch_size, num_samples)
            q_values = torch.min(q1_vals, q2_vals)
            best_idx = q_values.argmax(dim=1)
            
            # Select best actions
            batch_idx = torch.arange(batch_size).to(self.device)
            best_actions = action_samples[batch_idx, best_idx]
            
            return best_actions
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks with mixed action space"""
        
        # Ensure actions have correct shape [batch_size, 2]
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            # Select best next actions
            next_actions = self.select_action(next_states, num_samples=20)
            
            next_q1 = self.q1_target(next_states, next_actions).squeeze()
            next_q2 = self.q2_target(next_states, next_actions).squeeze()
            next_q = torch.min(next_q1, next_q2)
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
            target_q = torch.clamp(target_q, -50, 50)  # Stability
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Sample random actions for CQL penalty
            num_samples = 10
            batch_size = states.shape[0]
            
            # Random VP1 (binary) and VP2 (continuous)
            random_vp1 = torch.randint(0, 2, (batch_size, num_samples, 1)).float().to(self.device)
            random_vp2 = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            
            # Compute Q-values for random actions (batched)
            random_actions = torch.cat([random_vp1, random_vp2], dim=-1)  # [batch, samples, 2]
            
            # Reshape for batch processing
            states_expanded = states.unsqueeze(1).expand(-1, num_samples, -1)  # [batch, samples, state_dim]
            states_flat = states_expanded.reshape(-1, states.shape[-1])  # [batch*samples, state_dim]
            actions_flat = random_actions.reshape(-1, 2)  # [batch*samples, 2]
            
            # Single forward pass for all random actions
            random_q1 = self.q1(states_flat, actions_flat).reshape(batch_size, num_samples)
            
            # Conservative penalty with logsumexp
            logsumexp_q1 = torch.logsumexp(random_q1 / 10.0, dim=1) * 10.0
            cql1_loss = (logsumexp_q1 - current_q1).mean()
        else:
            cql1_loss = torch.tensor(0.0)
        
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2 (similar process)
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        if self.alpha > 0:
            # Reuse the same random actions and states_flat from Q1
            # Single forward pass for all random actions
            random_q2 = self.q2(states_flat, actions_flat).reshape(batch_size, num_samples)
            logsumexp_q2 = torch.logsumexp(random_q2 / 10.0, dim=1) * 10.0
            cql2_loss = (logsumexp_q2 - current_q2).mean()
        else:
            cql2_loss = torch.tensor(0.0)
        
        total_q2_loss = q2_loss + self.alpha * cql2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if isinstance(cql1_loss, torch.Tensor) else cql1_loss,
            'cql2_loss': cql2_loss.item() if isinstance(cql2_loss, torch.Tensor) else cql2_loss
        }


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(data_path: str, model_type: str = 'binary') -> Dict:
    """
    Prepare data for training.
    
    Args:
        data_path: Path to CSV file
        model_type: 'binary' or 'dual'
    """
    print(f"Loading data for {model_type} model...")
    df = pd.read_csv(data_path)
    df = df.sort_values(['subject_id', 'time_hour'])
    
    # State features
    state_features = ['mbp', 'lactate', 'sofa', 'uo_h', 'creatinine', 'bun', 'ventil', 'rrt']
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    
    unique_patients = df['subject_id'].unique()
    
    for patient_id in unique_patients:
        patient_data = df[df['subject_id'] == patient_id]
        
        if len(patient_data) < 5:
            continue
        
        states = patient_data[state_features].fillna(0).values
        
        if model_type == 'binary':
            # Binary action: VP1 only
            actions = patient_data['action_vaso'].values.astype(float).reshape(-1, 1)
        else:  # dual
            # VP1: binary, VP2: continuous [0, 0.5]
            vp1 = patient_data['action_vaso'].values.astype(float)
            vp2 = np.clip(patient_data.get('norepinephrine', pd.Series(np.zeros(len(patient_data)))).values / 0.5, 0, 1) * 0.5
            actions = np.column_stack([vp1, vp2])
        
        mortality = patient_data['death'].iloc[-1]
        
        for t in range(len(states) - 1):
            all_states.append(states[t])
            all_actions.append(actions[t])
            all_next_states.append(states[t + 1])
            
            # Simple reward function
            is_terminal = (t == len(states) - 2)
            reward = 0.0
            
            # Blood pressure reward
            next_mbp = states[t + 1][0]
            if 65 <= next_mbp <= 85:
                reward += 1.0
            elif next_mbp < 60:
                reward -= 2.0
            
            # Terminal reward
            if is_terminal:
                if mortality == 0:
                    reward += 10.0
                else:
                    reward -= 10.0
            
            all_rewards.append(reward)
            all_dones.append(1.0 if is_terminal else 0.0)
    
    # Convert to arrays and normalize
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=np.float32)
    
    # Normalize states
    scaler = StandardScaler()
    all_states = scaler.fit_transform(all_states)
    all_next_states = scaler.transform(all_next_states)
    
    print(f"Total transitions: {len(all_states)}")
    print(f"State shape: {all_states.shape}")
    print(f"Action shape: {all_actions.shape}")
    print(f"Reward range: [{all_rewards.min():.2f}, {all_rewards.max():.2f}]")
    
    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'next_states': all_next_states,
        'dones': all_dones,
        'scaler': scaler,
        'state_features': state_features
    }


def train_cql_model(
    model_type: str,
    alpha: float,
    data: Dict,
    epochs: int = 100,
    batch_size: int = 128,
    tau: float = 0.8,
    lr: float = 1e-3,
    save_dir: str = 'experiment'
) -> Tuple:
    """
    Train a CQL model (binary or dual).
    
    Args:
        model_type: 'binary' or 'dual'
        alpha: CQL penalty coefficient
        data: Prepared data dictionary
        epochs: Number of training epochs
        batch_size: Batch size for training
        tau: Soft update parameter
        lr: Learning rate
        save_dir: Directory to save models
    """
    print(f"\n{'='*70}")
    print(f" Training {model_type.upper()} CQL with alpha={alpha}")
    print(f"{'='*70}")
    
    # Split data (70/15/15)
    n_samples = len(data['states'])
    indices = np.arange(n_samples)
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Convert to tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_states = torch.FloatTensor(data['states'][train_idx]).to(device)
    train_actions = torch.FloatTensor(data['actions'][train_idx]).to(device)
    train_rewards = torch.FloatTensor(data['rewards'][train_idx]).to(device)
    train_next_states = torch.FloatTensor(data['next_states'][train_idx]).to(device)
    train_dones = torch.FloatTensor(data['dones'][train_idx]).to(device)
    
    val_states = torch.FloatTensor(data['states'][val_idx]).to(device)
    val_actions = torch.FloatTensor(data['actions'][val_idx]).to(device)
    
    # Initialize model
    state_dim = data['states'].shape[1]
    
    if model_type == 'binary':
        agent = BinaryCQL(
            state_dim=state_dim,
            alpha=alpha,
            gamma=0.95,
            tau=tau,
            lr=lr,
            grad_clip=1.0,
            device=device
        )
    else:  # dual
        agent = DualMixedCQL(
            state_dim=state_dim,
            alpha=alpha,
            gamma=0.95,
            tau=tau,
            lr=lr,
            grad_clip=1.0,
            device=device
        )
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    best_val_q = -float('inf')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        agent.q1.train()
        agent.q2.train()
        
        train_metrics = {'q1_loss': 0, 'cql1_loss': 0}
        n_batches = len(train_states) // batch_size
        
        # Shuffle training data
        perm = torch.randperm(len(train_states))
        
        for i in range(n_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            
            batch_states = train_states[batch_idx]
            batch_actions = train_actions[batch_idx]
            batch_rewards = train_rewards[batch_idx]
            batch_next_states = train_next_states[batch_idx]
            batch_dones = train_dones[batch_idx]
            
            metrics = agent.update(
                batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones
            )
            
            for key in train_metrics:
                train_metrics[key] += metrics.get(key, 0)
        
        for key in train_metrics:
            train_metrics[key] /= n_batches
        
        # Validation
        agent.q1.eval()
        agent.q2.eval()
        
        with torch.no_grad():
            val_q1 = agent.q1(val_states, val_actions).squeeze()
            val_q2 = agent.q2(val_states, val_actions).squeeze()
            val_q = torch.min(val_q1, val_q2).mean().item()
        
        # Save best model
        if val_q > best_val_q:
            best_val_q = val_q
            save_path = os.path.join(save_dir, f'{model_type}_rev_cql_alpha{alpha:.4f}_best.pt')
            if model_type == 'binary':
                agent.save(save_path)
            else:
                torch.save({
                    'q1_state_dict': agent.q1.state_dict(),
                    'q2_state_dict': agent.q2.state_dict(),
                    'q1_target_state_dict': agent.q1_target.state_dict(),
                    'q2_target_state_dict': agent.q2_target.state_dict(),
                    'alpha': alpha,
                    'state_dim': state_dim
                }, save_path)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}: Q1 Loss={train_metrics['q1_loss']:.4f}, "
                  f"CQL Loss={train_metrics['cql1_loss']:.4f}, "
                  f"Val Q={val_q:.4f}, Time={elapsed/60:.1f}min")
    
    # Save final model
    final_path = os.path.join(save_dir, f'{model_type}_cql_alpha{alpha:.3f}_final.pt')
    if model_type == 'binary':
        agent.save(final_path)
    else:
        torch.save({
            'q1_state_dict': agent.q1.state_dict(),
            'q2_state_dict': agent.q2.state_dict(),
            'q1_target_state_dict': agent.q1_target.state_dict(),
            'q2_target_state_dict': agent.q2_target.state_dict(),
            'alpha': alpha,
            'state_dim': state_dim
        }, final_path)
    
    total_time = time.time() - start_time
    print(f"\n✅ {model_type.upper()} CQL (alpha={alpha}) completed in {total_time/60:.1f} minutes!")
    print(f"Best validation Q-value: {best_val_q:.4f}")
    
    return agent, best_val_q


def main():
    """Main training function for all configurations"""
    print("="*70)
    print(" DUAL MIXED CQL TRAINING - ALL ALPHAS")
    print("="*70)
    print("\nThis script trains:")
    print("Dual CQL: VP1 binary, VP2 continuous [0, 0.5]")
    print("With alpha values: 0.0, 0.001, 0.01")
    print("Learning rate: 0.001, Tau: 0.8, Epochs: 100")
    
    # Alpha values to test
    alphas = [0.0, 0.001, 0.01]
    
    # Results storage
    results = {}
    
    # Skip Binary CQL - already trained
    print("\n" + "="*70)
    print(" SKIPPING BINARY CQL (Already Trained)")
    print("="*70)
    print("Binary models already exist:")
    print("  ✅ binary_rev_cql_alpha0.0000_best.pt")
    print("  ✅ binary_rev_cql_alpha0.0010_best.pt")
    print("  ✅ binary_rev_cql_alpha0.0100_best.pt")
    
    # Train Dual CQL (VP1 binary, VP2 continuous)
    print("\n" + "="*70)
    print(" DUAL MIXED CQL TRAINING (VP1 Binary, VP2 Continuous)")
    print("="*70)
    
    dual_data = prepare_data('sample_data_oviss.csv', model_type='dual')
    
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f" Training DUAL MIXED CQL with alpha={alpha}")
        print(f"{'='*70}")
        
        agent, best_val_q = train_cql_model(
            model_type='dual',
            alpha=alpha,
            data=dual_data,
            epochs=100,
            batch_size=128,
            tau=0.8,
            lr=0.001
        )
        results[f'dual_alpha{alpha}'] = best_val_q
    
    # Print summary
    print("\n" + "="*70)
    print(" TRAINING SUMMARY")
    print("="*70)
    print("\nDual Mixed CQL Best Validation Q-values:")
    print("-"*40)
    print(f"{'Model':<20} {'Alpha':<10} {'Best Val Q':<15}")
    print("-"*40)
    
    for key, val_q in results.items():
        alpha = float(key.split('alpha')[1])
        print(f"{'Dual Mixed CQL':<20} {alpha:<10.3f} {val_q:<15.4f}")
    
    print("\n✅ All training completed successfully!")
    print("Models saved in experiment/ directory")


if __name__ == "__main__":
    main()