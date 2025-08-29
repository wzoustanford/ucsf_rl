#!/usr/bin/env python3
"""
Stepwise action space formulation for VP2 with CQL training.

VP1: Binary (0 or 1) 
VP2: Stepwise changes from current dose
     - Decrease by 0.1 mcg/kg/min
     - Decrease by 0.05 mcg/kg/min
     - No change
     - Increase by 0.05 mcg/kg/min
     - Increase by 0.1 mcg/kg/min

Combined discrete action space: 2 (VP1) × 5 (VP2 changes) = 10 actions
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
from typing import Dict, Tuple, Optional, List
import json

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Stepwise Action Space Definition
# ============================================================================

class StepwiseActionSpace:
    """Manages stepwise action space for VP2"""
    
    # VP2 change actions
    VP2_CHANGES = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    VP2_MIN = 0.0
    VP2_MAX = 0.5
    
    def __init__(self):
        self.n_vp1_actions = 2  # Binary: 0 or 1
        self.n_vp2_actions = len(self.VP2_CHANGES)
        self.n_actions = self.n_vp1_actions * self.n_vp2_actions
        
        # Create action mapping
        self.action_map = []
        for vp1 in range(self.n_vp1_actions):
            for vp2_idx in range(self.n_vp2_actions):
                self.action_map.append((vp1, vp2_idx))
    
    def get_discrete_action(self, vp1: int, vp2_change_idx: int) -> int:
        """Get discrete action index from VP1 and VP2 change index"""
        return vp1 * self.n_vp2_actions + vp2_change_idx
    
    def decode_action(self, action_idx: int) -> Tuple[int, int]:
        """Decode discrete action to VP1 and VP2 change index"""
        return self.action_map[action_idx]
    
    def apply_action(self, action_idx: int, current_vp2: float) -> Tuple[float, float]:
        """
        Apply discrete action and return continuous VP1 and VP2 values
        
        Args:
            action_idx: Discrete action index (0-9)
            current_vp2: Current VP2 dose
        
        Returns:
            (vp1_value, new_vp2_value)
        """
        vp1, vp2_change_idx = self.decode_action(action_idx)
        vp2_change = self.VP2_CHANGES[vp2_change_idx]
        
        # Strict bounds checking - ensure VP2 stays within [0, 0.5]
        new_vp2 = current_vp2 + vp2_change
        new_vp2 = max(self.VP2_MIN, min(new_vp2, self.VP2_MAX))  # Clamp to [0, 0.5]
        
        return float(vp1), new_vp2
    
    def get_valid_actions(self, current_vp2: float, mask_invalid: bool = True) -> np.ndarray:
        """
        Get valid actions given current VP2 dose
        
        Args:
            current_vp2: Current VP2 dose
            mask_invalid: If True, return mask; if False, return valid indices
        
        Returns:
            Boolean mask or indices of valid actions
        """
        valid = np.ones(self.n_actions, dtype=bool)
        
        for action_idx in range(self.n_actions):
            vp1, vp2_change_idx = self.decode_action(action_idx)
            vp2_change = self.VP2_CHANGES[vp2_change_idx]
            new_vp2 = current_vp2 + vp2_change
            
            # Mark invalid if resulting dose would be out of bounds [0, 0.5]
            # Use small epsilon for numerical stability
            eps = 1e-6
            if new_vp2 < (self.VP2_MIN - eps) or new_vp2 > (self.VP2_MAX + eps):
                valid[action_idx] = False
        
        if mask_invalid:
            return valid
        else:
            return np.where(valid)[0]


# ============================================================================
# Q-Network for Discrete Actions
# ============================================================================

class DiscreteQNetwork(nn.Module):
    """Q-network for discrete actions Q(s) -> Q-values for all actions"""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, n_actions)
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            q_values: [batch_size, n_actions]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# Stepwise CQL Implementation
# ============================================================================

class StepwiseCQL:
    """CQL with stepwise action space for VP2"""
    
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
        self.action_space = StepwiseActionSpace()
        self.n_actions = self.action_space.n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Extended state dimension (includes current VP2 dose)
        self.extended_state_dim = state_dim + 1
        
        # Q-networks
        self.q1 = DiscreteQNetwork(self.extended_state_dim, self.n_actions).to(device)
        self.q2 = DiscreteQNetwork(self.extended_state_dim, self.n_actions).to(device)
        self.q1_target = DiscreteQNetwork(self.extended_state_dim, self.n_actions).to(device)
        self.q2_target = DiscreteQNetwork(self.extended_state_dim, self.n_actions).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def extend_state(self, states: torch.Tensor, current_vp2: torch.Tensor) -> torch.Tensor:
        """Add current VP2 dose to state"""
        if current_vp2.dim() == 1:
            current_vp2 = current_vp2.unsqueeze(1)
        return torch.cat([states, current_vp2], dim=-1)
    
    def get_valid_action_masks(self, current_vp2_batch: torch.Tensor) -> torch.Tensor:
        """Get valid action masks for a batch of states"""
        batch_size = current_vp2_batch.shape[0]
        masks = torch.zeros(batch_size, self.n_actions, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            # Ensure VP2 is within valid range before checking valid actions
            vp2_val = torch.clamp(current_vp2_batch[i], 0.0, 0.5).item()
            valid = self.action_space.get_valid_actions(vp2_val, mask_invalid=True)
            masks[i] = torch.tensor(valid, device=self.device)
        
        return masks
    
    def select_action(self, state: torch.Tensor, current_vp2: torch.Tensor) -> torch.Tensor:
        """Select best valid action for given state"""
        with torch.no_grad():
            extended_state = self.extend_state(state, current_vp2)
            
            # Get Q-values
            q1_vals = self.q1(extended_state)
            q2_vals = self.q2(extended_state)
            q_vals = torch.min(q1_vals, q2_vals)
            
            # Apply valid action mask
            valid_masks = self.get_valid_action_masks(current_vp2)
            q_vals[~valid_masks] = -float('inf')
            
            # Select best action
            best_actions = q_vals.argmax(dim=1)
            return best_actions
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        current_vp2: torch.Tensor,
        next_vp2: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks"""
        
        # Ensure VP2 values are within valid bounds [0, 0.5]
        current_vp2 = torch.clamp(current_vp2, 0.0, 0.5)
        next_vp2 = torch.clamp(next_vp2, 0.0, 0.5)
        
        # Extend states with VP2 information
        extended_states = self.extend_state(states, current_vp2)
        extended_next_states = self.extend_state(next_states, next_vp2)
        
        # Convert actions to long tensor for indexing
        actions = actions.long()
        
        # Compute target Q-values
        with torch.no_grad():
            # Get Q-values for all actions at next state
            next_q1 = self.q1_target(extended_next_states)
            next_q2 = self.q2_target(extended_next_states)
            
            # Apply valid action masks
            next_valid_masks = self.get_valid_action_masks(next_vp2)
            next_q1[~next_valid_masks] = -float('inf')
            next_q2[~next_valid_masks] = -float('inf')
            
            # Take min over networks, then max over actions
            next_q = torch.min(next_q1, next_q2)
            next_q_max, _ = next_q.max(dim=1)
            
            target_q = rewards + self.gamma * next_q_max * (1 - dones)
            target_q = torch.clamp(target_q, -50, 50)  # Stability
        
        # Update Q1
        current_q1_all = self.q1(extended_states)
        current_q1 = current_q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Get valid action masks
            valid_masks = self.get_valid_action_masks(current_vp2)
            
            # Mask invalid actions
            masked_q1 = current_q1_all.clone()
            masked_q1[~valid_masks] = -1e8
            
            # Conservative penalty
            logsumexp_q1 = torch.logsumexp(masked_q1 / 10.0, dim=1) * 10.0
            cql1_loss = (logsumexp_q1 - current_q1).mean()
        else:
            cql1_loss = torch.tensor(0.0)
        
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2 (similar process)
        current_q2_all = self.q2(extended_states)
        current_q2 = current_q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        if self.alpha > 0:
            masked_q2 = current_q2_all.clone()
            masked_q2[~valid_masks] = -1e8
            logsumexp_q2 = torch.logsumexp(masked_q2 / 10.0, dim=1) * 10.0
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
            'tau': self.tau,
            'state_dim': self.state_dim
        }, path)


# ============================================================================
# Data Preparation for Stepwise Actions
# ============================================================================

def prepare_stepwise_data(data_path: str) -> Dict:
    """
    Prepare data for stepwise action training.
    Actions are encoded as discrete indices based on VP1 and VP2 changes.
    """
    print("Loading data for stepwise action model...")
    df = pd.read_csv(data_path)
    df = df.sort_values(['subject_id', 'time_hour'])
    
    # State features
    state_features = ['mbp', 'lactate', 'sofa', 'uo_h', 'creatinine', 'bun', 'ventil', 'rrt']
    
    action_space = StepwiseActionSpace()
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_current_vp2 = []
    all_next_vp2 = []
    
    unique_patients = df['subject_id'].unique()
    
    for patient_id in unique_patients:
        patient_data = df[df['subject_id'] == patient_id]
        
        if len(patient_data) < 5:
            continue
        
        states = patient_data[state_features].fillna(0).values
        
        # Get VP1 actions
        vp1_actions = patient_data['action_vaso'].values.astype(float)
        
        # Get VP2 doses (using norepinephrine as proxy, scaled to 0-0.5 range)
        vp2_doses = patient_data.get('norepinephrine', pd.Series(np.zeros(len(patient_data)))).values
        vp2_doses = np.clip(vp2_doses, 0, 0.5)  # Ensure in valid range
        
        mortality = patient_data['death'].iloc[-1]
        
        for t in range(len(states) - 1):
            # Current and next VP2 doses - ensure they're within bounds
            current_vp2 = np.clip(vp2_doses[t], 0.0, 0.5)
            next_vp2 = np.clip(vp2_doses[t + 1], 0.0, 0.5)
            
            # Determine VP2 change action
            vp2_change = next_vp2 - current_vp2
            
            # Find closest matching action
            vp2_change_idx = np.argmin(np.abs(action_space.VP2_CHANGES - vp2_change))
            
            # Get discrete action index
            vp1 = int(vp1_actions[t])
            action_idx = action_space.get_discrete_action(vp1, vp2_change_idx)
            
            all_states.append(states[t])
            all_actions.append(action_idx)
            all_next_states.append(states[t + 1])
            all_current_vp2.append(current_vp2)
            all_next_vp2.append(next_vp2)
            
            # Reward function
            is_terminal = (t == len(states) - 2)
            reward = 0.0
            
            # Blood pressure reward
            next_mbp = states[t + 1][0]
            if 65 <= next_mbp <= 85:
                reward += 1.0
            elif next_mbp < 60:
                reward -= 2.0
            
            # Penalize excessive VP2 changes
            if abs(vp2_change) > 0.05:
                reward -= 0.5 * abs(vp2_change) / 0.1
            
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
    all_actions = np.array(all_actions, dtype=np.int64)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=np.float32)
    all_current_vp2 = np.array(all_current_vp2, dtype=np.float32)
    all_next_vp2 = np.array(all_next_vp2, dtype=np.float32)
    
    # Normalize states
    scaler = StandardScaler()
    all_states = scaler.fit_transform(all_states)
    all_next_states = scaler.transform(all_next_states)
    
    print(f"Total transitions: {len(all_states)}")
    print(f"State shape: {all_states.shape}")
    print(f"Action distribution: {np.bincount(all_actions)}")
    print(f"VP2 range: [{all_current_vp2.min():.3f}, {all_current_vp2.max():.3f}]")
    print(f"Reward range: [{all_rewards.min():.2f}, {all_rewards.max():.2f}]")
    
    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'next_states': all_next_states,
        'dones': all_dones,
        'current_vp2': all_current_vp2,
        'next_vp2': all_next_vp2,
        'scaler': scaler,
        'state_features': state_features
    }


def train_stepwise_cql(
    alpha: float,
    data: Dict,
    epochs: int = 100,
    batch_size: int = 128,
    tau: float = 0.005,
    lr: float = 3e-4,
    save_dir: str = 'experiment'
) -> Tuple:
    """Train stepwise CQL model"""
    print(f"\n{'='*70}")
    print(f" Training STEPWISE CQL with alpha={alpha}")
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
    train_actions = torch.LongTensor(data['actions'][train_idx]).to(device)
    train_rewards = torch.FloatTensor(data['rewards'][train_idx]).to(device)
    train_next_states = torch.FloatTensor(data['next_states'][train_idx]).to(device)
    train_dones = torch.FloatTensor(data['dones'][train_idx]).to(device)
    train_current_vp2 = torch.FloatTensor(data['current_vp2'][train_idx]).to(device)
    train_next_vp2 = torch.FloatTensor(data['next_vp2'][train_idx]).to(device)
    
    val_states = torch.FloatTensor(data['states'][val_idx]).to(device)
    val_actions = torch.LongTensor(data['actions'][val_idx]).to(device)
    val_current_vp2 = torch.FloatTensor(data['current_vp2'][val_idx]).to(device)
    
    # Initialize model
    state_dim = data['states'].shape[1]
    
    agent = StepwiseCQL(
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
    
    training_history = []
    
    for epoch in range(epochs):
        # Training
        agent.q1.train()
        agent.q2.train()
        
        train_metrics = {'q1_loss': 0, 'cql1_loss': 0, 'total_q1_loss': 0}
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
            batch_current_vp2 = train_current_vp2[batch_idx]
            batch_next_vp2 = train_next_vp2[batch_idx]
            
            metrics = agent.update(
                batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones,
                batch_current_vp2, batch_next_vp2
            )
            
            for key in train_metrics:
                train_metrics[key] += metrics.get(key, 0)
        
        for key in train_metrics:
            train_metrics[key] /= n_batches
        
        # Validation
        agent.q1.eval()
        agent.q2.eval()
        
        with torch.no_grad():
            extended_val_states = agent.extend_state(val_states, val_current_vp2)
            val_q1_all = agent.q1(extended_val_states)
            val_q2_all = agent.q2(extended_val_states)
            
            # Get Q-values for actual actions taken
            val_q1 = val_q1_all.gather(1, val_actions.unsqueeze(1)).squeeze()
            val_q2 = val_q2_all.gather(1, val_actions.unsqueeze(1)).squeeze()
            val_q = torch.min(val_q1, val_q2).mean().item()
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_q1_loss': train_metrics['q1_loss'],
            'train_cql1_loss': train_metrics['cql1_loss'],
            'val_q': val_q
        })
        
        # Save best model
        if val_q > best_val_q:
            best_val_q = val_q
            save_path = os.path.join(save_dir, f'stepwise_cql_alpha{alpha:.4f}_best.pt')
            agent.save(save_path)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}: Q1 Loss={train_metrics['q1_loss']:.4f}, "
                  f"CQL Loss={train_metrics['cql1_loss']:.4f}, "
                  f"Val Q={val_q:.4f}, Time={elapsed/60:.1f}min")
    
    # Save final model
    final_path = os.path.join(save_dir, f'stepwise_cql_alpha{alpha:.3f}_final.pt')
    agent.save(final_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f'stepwise_cql_alpha{alpha:.3f}_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n✅ STEPWISE CQL (alpha={alpha}) completed in {total_time/60:.1f} minutes!")
    print(f"Best validation Q-value: {best_val_q:.4f}")
    
    return agent, best_val_q


def main():
    """Main training function for stepwise action CQL"""
    print("="*70)
    print(" STEPWISE ACTION CQL TRAINING")
    print("="*70)
    print("\nAction Space:")
    print("- VP1: Binary (0 or 1)")
    print("- VP2: Stepwise changes:")
    print("  * -0.1 mcg/kg/min")
    print("  * -0.05 mcg/kg/min")
    print("  * No change")
    print("  * +0.05 mcg/kg/min")
    print("  * +0.1 mcg/kg/min")
    print(f"\nTotal discrete actions: 10 (2 VP1 × 5 VP2 changes)")
    
    # Alpha values to test
    alphas = [0.0, 0.001, 0.01]
    
    # Results storage
    results = {}
    
    # Prepare data
    data = prepare_stepwise_data('sample_data_oviss.csv')
    
    # Train models with different alpha values
    for alpha in alphas:
        agent, best_val_q = train_stepwise_cql(
            alpha=alpha,
            data=data,
            epochs=100,
            batch_size=128,
            tau=0.005,
            lr=3e-4
        )
        results[f'alpha{alpha}'] = best_val_q
    
    # Print summary
    print("\n" + "="*70)
    print(" TRAINING SUMMARY")
    print("="*70)
    print("\nBest Validation Q-values:")
    print("-"*40)
    print(f"{'Alpha':<10} {'Best Val Q':<15}")
    print("-"*40)
    
    for key, val_q in results.items():
        alpha = float(key.split('alpha')[1])
        print(f"{alpha:<10.4f} {val_q:<15.4f}")
    
    print("\n✅ All training completed successfully!")
    print("Models saved in experiment/ directory")
    
    # Compare with continuous baseline if available
    print("\n" + "="*70)
    print(" ACTION SPACE COMPARISON")
    print("="*70)
    print("\nStepwise actions provide:")
    print("1. Clinically interpretable dose adjustments")
    print("2. Smoother transitions between doses")
    print("3. Built-in safety constraints (bounds checking)")
    print("4. More realistic clinical decision making")


if __name__ == "__main__":
    main()