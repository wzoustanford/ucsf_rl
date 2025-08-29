"""
Stable CQL training with gradient clipping and better hyperparameters
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict
import os


class ContinuousQNetwork(nn.Module):
    """Q-network for continuous actions Q(s,a)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = state_dim + action_dim
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
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class StableCQL:
    """CQL with stability improvements"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        alpha: float = 2.0,  # Moderate CQL penalty
        gamma: float = 0.95,  # Slightly lower discount
        tau: float = 0.005,
        lr: float = 1e-4,  # Lower learning rate
        grad_clip: float = 1.0,  # Gradient clipping
        num_iterations_q1_optimize_per_batch: int = 200,  # Number of Q1 optimization iterations per batch
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks
        self.q1 = ContinuousQNetwork(state_dim, action_dim).to(device)
        self.q2 = ContinuousQNetwork(state_dim, action_dim).to(device)
        self.q1_target = ContinuousQNetwork(state_dim, action_dim).to(device)
        self.q2_target = ContinuousQNetwork(state_dim, action_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers with weight decay
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr, weight_decay=1e-5)
        
        # Number of iterations to run per batch for Q1 optimization
        self.num_iterations_q1_optimize_per_batch = num_iterations_q1_optimize_per_batch 
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Select action with optional epsilon-greedy exploration"""
        if np.random.random() < epsilon:
            # Random action
            vp1 = np.random.random()
            vp2 = np.random.random() * 0.5
            return np.array([vp1, vp2])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            num_samples = 50
            state_expanded = state_tensor.expand(num_samples, -1)
            
            # Sample candidate actions
            vp1_samples = torch.rand(num_samples, 1).to(self.device)
            vp2_samples = torch.rand(num_samples, 1).to(self.device) * 0.5
            action_samples = torch.cat([vp1_samples, vp2_samples], dim=1)
            
            # Evaluate Q-values
            q1_values = self.q1(state_expanded, action_samples).squeeze()
            q2_values = self.q2(state_expanded, action_samples).squeeze()
            q_values = torch.min(q1_values, q2_values)
            
            # Select best action
            best_idx = q_values.argmax()
            best_action = action_samples[best_idx].cpu().numpy()
            
            return best_action
    
    def select_actions_batch(self, states: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        """Select actions for a batch of states efficiently"""
        batch_size = states.shape[0]
        
        with torch.no_grad():
            # Expand states for all samples at once
            states_expanded = states.unsqueeze(1).expand(-1, num_samples, -1)  # [batch, samples, state_dim]
            states_flat = states_expanded.reshape(-1, states.shape[-1])  # [batch*samples, state_dim]
            
            # Sample all actions at once
            vp1_samples = torch.rand(batch_size, num_samples, 1).to(self.device)
            vp2_samples = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            actions_samples = torch.cat([vp1_samples, vp2_samples], dim=-1)  # [batch, samples, 2]
            actions_flat = actions_samples.reshape(-1, 2)  # [batch*samples, 2]
            
            # Evaluate Q-values for all samples at once (single forward pass!)
            q1_values = self.q1(states_flat, actions_flat).reshape(batch_size, num_samples)
            q2_values = self.q2(states_flat, actions_flat).reshape(batch_size, num_samples)
            q_values = torch.min(q1_values, q2_values)
            
            # Select best action for each state in batch
            best_indices = q_values.argmax(dim=1)  # [batch]
            batch_indices = torch.arange(batch_size).to(self.device)
            best_actions = actions_samples[batch_indices, best_indices]  # [batch, 2]
            
            return best_actions
    
    def compute_cql_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        q_network: nn.Module,
        num_samples: int = 10
    ) -> torch.Tensor:
        """Compute CQL penalty with stability improvements"""
        batch_size = states.shape[0]
        
        # Current Q-values
        current_q = q_network(states, actions).squeeze()
        
        # Sample random actions for CQL penalty
        with torch.no_grad():
            vp1_random = torch.rand(batch_size, num_samples, 1).to(self.device)
            vp2_random = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            random_actions = torch.cat([vp1_random, vp2_random], dim=-1)
        
        # Compute Q-values for random actions
        random_q_list = []
        for i in range(num_samples):
            q_val = q_network(states, random_actions[:, i, :]).squeeze()
            random_q_list.append(q_val.unsqueeze(1))
        
        random_q = torch.cat(random_q_list, dim=1)
        
        # CQL penalty with numerical stability
        logsumexp = torch.logsumexp(random_q / 10.0, dim=1) * 10.0  # Temperature scaling
        cql_loss = (logsumexp - current_q).mean()
        
        return cql_loss
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks with stability improvements"""
        
        # Clip rewards for stability
        rewards = torch.clamp(rewards, -10, 10)
        
        # Compute target Q-values
        with torch.no_grad():
            # FAST: Use batched action selection (single forward pass instead of 128!)
            next_actions = self.select_actions_batch(next_states)
            
            next_q1 = self.q1_target(next_states, next_actions).squeeze()
            next_q2 = self.q2_target(next_states, next_actions).squeeze()
            next_q = torch.min(next_q1, next_q2)
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
            target_q = torch.clamp(target_q, -50, 50)  # Clip targets
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        cql1_loss = self.compute_cql_loss(states, actions, self.q1)
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        cql2_loss = self.compute_cql_loss(states, actions, self.q2)
        total_q2_loss = q2_loss + self.alpha * cql2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item(),
            'cql2_loss': cql2_loss.item()
        }


def compute_outcome_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    action: np.ndarray,
    is_terminal: bool,
    mortality: int,
    state_features: list
) -> float:
    """Simplified outcome-based reward"""
    reward = 0.0
    
    # Get indices
    mbp_idx = state_features.index('mbp')
    lactate_idx = state_features.index('lactate')
    
    mbp = state[mbp_idx]
    next_mbp = next_state[mbp_idx]
    lactate = state[lactate_idx]
    next_lactate = next_state[lactate_idx]
    
    vp1_dose = action[0]
    vp2_dose = action[1]
    
    # Blood pressure reward
    if 65 <= next_mbp <= 85:
        reward += 1.0
    elif next_mbp < 60:
        reward -= 2.0
    elif next_mbp > 100:
        reward -= 0.5
    
    # Lactate improvement
    if next_lactate < lactate - 0.1:
        reward += 0.5
    
    # Minimize vasopressor (small penalty)
    total_vaso = vp1_dose + vp2_dose * 2
    if total_vaso > 1.0:
        reward -= 0.2 * (total_vaso - 1.0)
    
    # Terminal rewards
    if is_terminal:
        if mortality == 0:
            reward += 10.0
        else:
            reward -= 10.0
    
    return reward


def prepare_data(data_path: str) -> Dict:
    """Prepare data with simplified rewards"""
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.sort_values(['subject_id', 'time_hour'])
    
    state_features = ['mbp', 'lactate', 'sofa', 'uo_h', 'creatinine', 'bun', 'ventil', 'rrt']
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    patient_id_map = []
    
    unique_patients = df['subject_id'].unique()  # Use ALL patients
    
    for patient_id in unique_patients:
        patient_data = df[df['subject_id'] == patient_id]
        
        if len(patient_data) < 5:
            continue
        
        states = patient_data[state_features].fillna(0).values
        vp1 = patient_data['action_vaso'].values.astype(float)
        vp2 = np.clip(patient_data['norepinephrine'].values / 0.5, 0, 1) * 0.5
        actions = np.column_stack([vp1, vp2])
        mortality = patient_data['death'].iloc[-1]
        
        for t in range(len(states) - 1):
            all_states.append(states[t])
            all_actions.append(actions[t])
            all_next_states.append(states[t + 1])
            patient_id_map.append(patient_id)
            
            is_terminal = (t == len(states) - 2)
            reward = compute_outcome_reward(
                states[t], states[t + 1], actions[t],
                is_terminal, mortality, state_features
            )
            
            all_rewards.append(reward)
            all_dones.append(1.0 if is_terminal else 0.0)
    
    # Convert and normalize
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=np.float32)
    patient_id_map = np.array(patient_id_map)
    
    scaler = StandardScaler()
    all_states = scaler.fit_transform(all_states)
    all_next_states = scaler.transform(all_next_states)
    
    print(f"Total transitions: {len(all_states)}")
    print(f"Reward range: [{all_rewards.min():.2f}, {all_rewards.max():.2f}]")
    
    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'next_states': all_next_states,
        'dones': all_dones,
        'patient_ids': patient_id_map,
        'scaler': scaler,
        'state_features': state_features
    }


def main():
    """Train stable CQL"""
    
    print("="*70)
    print("STABLE CQL TRAINING - NO CONCORDANCE REWARDS")
    print("="*70)
    
    # Prepare data
    data = prepare_data('sample_data_oviss.csv')
    
    # Split data with 70/15/15 train/val/test
    unique_patients = np.unique(data['patient_ids'])
    n_patients = len(unique_patients)
    
    # First split: 85% train+val, 15% test
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=0.15, random_state=42
    )
    
    # Second split: from 85%, take 15/85 for val (which gives 15% of total)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.15/0.85, random_state=42
    )
    
    train_idx = np.isin(data['patient_ids'], train_patients)
    val_idx = np.isin(data['patient_ids'], val_patients)
    test_idx = np.isin(data['patient_ids'], test_patients)
    
    print(f"\nPatient splits (70/15/15):")
    print(f"  Train: {len(train_patients)} ({len(train_patients)/n_patients*100:.1f}%)")
    print(f"  Val: {len(val_patients)} ({len(val_patients)/n_patients*100:.1f}%)")
    print(f"  Test: {len(test_patients)} ({len(test_patients)/n_patients*100:.1f}%)")
    
    # Initialize agent
    agent = StableCQL(
        state_dim=len(data['state_features']),
        action_dim=2,
        alpha=0.0,  # NO CONSERVATIVE PENALTY - testing standard Q-learning
        gamma=0.95,
        lr=1e-4,
        grad_clip=1.0
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(data['states'][train_idx]),
        torch.FloatTensor(data['actions'][train_idx]),
        torch.FloatTensor(data['rewards'][train_idx]),
        torch.FloatTensor(data['next_states'][train_idx]),
        torch.FloatTensor(data['dones'][train_idx])
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(data['states'][val_idx]),
        torch.FloatTensor(data['actions'][val_idx]),
        torch.FloatTensor(data['rewards'][val_idx]),
        torch.FloatTensor(data['next_states'][val_idx]),
        torch.FloatTensor(data['dones'][val_idx])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Training
    epochs = 100  # More epochs for larger dataset
    print(f"\nTraining for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        agent.q1.train()
        agent.q2.train()
        
        train_metrics = {'q1_loss': 0, 'cql1_loss': 0}
        
        for batch in train_loader:
            states, actions, rewards, next_states, dones = [b.to(agent.device) for b in batch]
            metrics = agent.update(states, actions, rewards, next_states, dones)
            train_metrics['q1_loss'] += metrics['q1_loss']
            train_metrics['cql1_loss'] += metrics['cql1_loss']
        
        train_metrics['q1_loss'] /= len(train_loader)
        train_metrics['cql1_loss'] /= len(train_loader)
        
        # Validation
        agent.q1.eval()
        agent.q2.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states, actions, rewards, next_states, dones = [b.to(agent.device) for b in batch]
                q1_val = agent.q1(states, actions).squeeze()
                q2_val = agent.q2(states, actions).squeeze()
                q_val = torch.min(q1_val, q2_val)
                val_loss += q_val.mean().item()
        
        val_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'scaler': data['scaler'],
                'state_features': data['state_features']
            }, 'experiment/cql_best_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Q-Loss={train_metrics['q1_loss']:.4f}, "
                  f"CQL-Loss={train_metrics['cql1_loss']:.4f}, Val-Loss={val_loss:.4f}")
    
    # Save model
    os.makedirs('experiment', exist_ok=True)
    torch.save({
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'scaler': data['scaler'],
        'state_features': data['state_features']
    }, 'experiment/cql_stable_no_concordance.pt')
    
    print("\n✅ Training completed!")
    print("Model saved to: experiment/cql_stable_no_concordance.pt")
    
    return agent, data


if __name__ == "__main__":
    agent, data = main()
