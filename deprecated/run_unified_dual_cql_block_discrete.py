"""
Unified CQL training with dual discrete action space for VP1 and VP2
VP1: Binary (0 or 1)
VP2: Discrete with bins (5, 10, 30, 50 options for 0-0.5 mcg/kg/min range)
Uses Q(s,a) architecture matching train_cql_stable.py for model alignment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime


class ContinuousQNetwork(nn.Module):
    """Q-network for continuous actions Q(s,a) - same as train_cql_stable.py"""
    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 128):
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


class DualDiscreteCQL:
    """CQL with dual discrete action spaces using Q(s,a) architecture"""
    
    def __init__(
        self,
        state_dim: int,
        vp2_bins: int = 10,  # Number of discrete bins for VP2
        alpha: float = 2.0,
        gamma: float = 0.95,
        tau: float = 0.8,
        lr: float = 0.001,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.vp1_actions = 2  # Binary: 0 or 1
        self.vp2_actions = vp2_bins
        self.vp2_bins = vp2_bins
        self.vp2_max = 0.5  # Maximum VP2 dose in mcg/kg/min
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks with same architecture as train_cql_stable.py
        self.q1 = ContinuousQNetwork(state_dim, action_dim=2).to(device)
        self.q2 = ContinuousQNetwork(state_dim, action_dim=2).to(device)
        self.q1_target = ContinuousQNetwork(state_dim, action_dim=2).to(device)
        self.q2_target = ContinuousQNetwork(state_dim, action_dim=2).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers with weight decay
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr, weight_decay=1e-5)
        
        # Precompute all possible discrete actions as continuous values
        self._precompute_action_space()
    
    def _precompute_action_space(self):
        """Precompute all possible discrete action combinations as continuous values"""
        self.all_actions = []
        for vp1 in range(self.vp1_actions):
            for vp2 in range(self.vp2_actions):
                vp1_cont, vp2_cont = self.discrete_to_continuous_action(vp1, vp2)
                self.all_actions.append([vp1_cont, vp2_cont])
        self.all_actions = torch.FloatTensor(self.all_actions).to(self.device)
        self.n_actions = len(self.all_actions)
    
    def continuous_to_discrete_action(self, vp1_continuous: np.ndarray, vp2_continuous: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert continuous actions to discrete indices"""
        # VP1: Binary threshold at 0.5
        vp1_discrete = (vp1_continuous >= 0.5).astype(int)
        
        # VP2: Discretize into bins
        vp2_normalized = np.clip(vp2_continuous / self.vp2_max, 0, 1)
        vp2_discrete = np.floor(vp2_normalized * self.vp2_bins).astype(int)
        vp2_discrete = np.clip(vp2_discrete, 0, self.vp2_bins - 1)
        
        return vp1_discrete, vp2_discrete
    
    def discrete_to_continuous_action(self, vp1_discrete: int, vp2_discrete: int) -> Tuple[float, float]:
        """Convert discrete actions back to continuous values"""
        vp1_continuous = float(vp1_discrete)
        vp2_continuous = (vp2_discrete + 0.5) / self.vp2_bins * self.vp2_max
        return vp1_continuous, vp2_continuous
    
    def discrete_to_continuous_batch(self, vp1_discrete: torch.Tensor, vp2_discrete: torch.Tensor) -> torch.Tensor:
        """Convert batch of discrete actions to continuous tensors"""
        vp1_continuous = vp1_discrete.float()
        vp2_continuous = (vp2_discrete.float() + 0.5) / self.vp2_bins * self.vp2_max
        return torch.stack([vp1_continuous, vp2_continuous], dim=-1)
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, int, np.ndarray]:
        """Select discrete action by evaluating Q(s,a) for all discrete actions"""
        if np.random.random() < epsilon:
            # Random discrete actions
            vp1_discrete = np.random.randint(0, self.vp1_actions)
            vp2_discrete = np.random.randint(0, self.vp2_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Expand state for all possible actions
                state_expanded = state_tensor.expand(self.n_actions, -1)
                
                # Evaluate Q(s,a) for all possible discrete actions
                q1_values = self.q1(state_expanded, self.all_actions).squeeze()
                q2_values = self.q2(state_expanded, self.all_actions).squeeze()
                q_values = torch.min(q1_values, q2_values)
                
                # Select best action
                best_idx = q_values.argmax().item()
                vp1_discrete = best_idx // self.vp2_actions
                vp2_discrete = best_idx % self.vp2_actions
        
        # Convert to continuous for compatibility
        vp1_cont, vp2_cont = self.discrete_to_continuous_action(vp1_discrete, vp2_discrete)
        
        return vp1_discrete, vp2_discrete, np.array([vp1_cont, vp2_cont])
    
    def select_actions_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Select best discrete actions for a batch of states - OPTIMIZED"""
        batch_size = states.shape[0]
        
        with torch.no_grad():
            # Expand states for all samples at once
            states_expanded = states.unsqueeze(1).expand(-1, self.n_actions, -1)  # [batch, n_actions, state_dim]
            states_flat = states_expanded.reshape(-1, states.shape[-1])  # [batch * n_actions, state_dim]
            
            # Expand actions for all states
            actions_expanded = self.all_actions.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_actions, 2]
            actions_flat = actions_expanded.reshape(-1, 2)  # [batch * n_actions, 2]
            
            # Evaluate Q-values for all state-action pairs at once
            q1_values = self.q1(states_flat, actions_flat).reshape(batch_size, self.n_actions)
            q2_values = self.q2(states_flat, actions_flat).reshape(batch_size, self.n_actions)
            q_values = torch.min(q1_values, q2_values)  # [batch_size, n_actions]
            
            # Select best action for each state in batch
            best_indices = q_values.argmax(dim=1)  # [batch_size]
            best_actions = self.all_actions[best_indices]  # [batch_size, 2]
            
            return best_actions
    
    def compute_cql_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        q_network: nn.Module,
        num_samples: int = 10
    ) -> torch.Tensor:
        """Compute CQL penalty by sampling from discrete action space"""
        batch_size = states.shape[0]
        
        # Current Q-values for taken actions
        current_q = q_network(states, actions).squeeze()
        
        # Sample random discrete actions and convert to continuous
        with torch.no_grad():
            random_actions = []
            for _ in range(num_samples):
                # Sample random discrete actions
                vp1_random = torch.randint(0, self.vp1_actions, (batch_size,)).to(self.device)
                vp2_random = torch.randint(0, self.vp2_actions, (batch_size,)).to(self.device)
                # Convert to continuous
                actions_cont = self.discrete_to_continuous_batch(vp1_random, vp2_random)
                random_actions.append(actions_cont)
        
        # Compute Q-values for random actions
        random_q_list = []
        for random_action in random_actions:
            q_val = q_network(states, random_action).squeeze()
            random_q_list.append(q_val.unsqueeze(1))
        
        random_q = torch.cat(random_q_list, dim=1)
        
        # CQL penalty with numerical stability
        logsumexp = torch.logsumexp(random_q / 10.0, dim=1) * 10.0  # Temperature scaling
        cql_loss = (logsumexp - current_q).mean()
        
        return cql_loss
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,  # Already continuous
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks"""
        
        # Clip rewards for stability
        rewards = torch.clamp(rewards, -10, 10)
        
        # Compute target Q-values
        with torch.no_grad():
            # Select best discrete actions for next states
            next_actions = self.select_actions_batch(next_states)
            
            next_q1 = self.q1_target(next_states, next_actions).squeeze()
            next_q2 = self.q2_target(next_states, next_actions).squeeze()
            next_q = torch.min(next_q1, next_q2)
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
            target_q = torch.clamp(target_q, -50, 50)
        
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
            'cql2_loss': cql2_loss.item(),
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item()
        }


def compute_outcome_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    action: np.ndarray,
    is_terminal: bool,
    mortality: int,
    state_features: list
) -> float:
    """Simplified outcome-based reward (same as original)"""
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


def prepare_data(data_path: str, vp2_bins: int) -> Dict:
    """Prepare data with discrete actions converted to continuous for Q(s,a)"""
    
    print(f"Loading data with VP2 bins={vp2_bins}...")
    df = pd.read_csv(data_path)
    df = df.sort_values(['subject_id', 'time_hour'])
    
    state_features = ['mbp', 'lactate', 'sofa', 'uo_h', 'creatinine', 'bun', 'ventil', 'rrt']
    
    all_states = []
    all_actions_continuous = []  # Store continuous representation for Q(s,a)
    all_rewards = []
    all_next_states = []
    all_dones = []
    patient_id_map = []
    
    unique_patients = df['subject_id'].unique()
    
    # Create temporary agent just for discretization
    temp_agent = DualDiscreteCQL(
        state_dim=len(state_features),
        vp2_bins=vp2_bins
    )
    
    for patient_id in unique_patients:
        patient_data = df[df['subject_id'] == patient_id]
        
        if len(patient_data) < 5:
            continue
        
        states = patient_data[state_features].fillna(0).values
        vp1_continuous = patient_data['action_vaso'].values.astype(float)
        vp2_continuous = np.clip(patient_data['norepinephrine'].values / 0.5, 0, 1) * 0.5
        mortality = patient_data['death'].iloc[-1]
        
        # Convert to discrete actions then back to continuous (for discretization)
        vp1_discrete, vp2_discrete = temp_agent.continuous_to_discrete_action(
            vp1_continuous, vp2_continuous
        )
        
        # Convert discrete back to continuous for Q(s,a) input
        actions_continuous = []
        for vp1_d, vp2_d in zip(vp1_discrete, vp2_discrete):
            vp1_c, vp2_c = temp_agent.discrete_to_continuous_action(vp1_d, vp2_d)
            actions_continuous.append([vp1_c, vp2_c])
        actions_continuous = np.array(actions_continuous)
        
        for t in range(len(states) - 1):
            all_states.append(states[t])
            all_actions_continuous.append(actions_continuous[t])
            all_next_states.append(states[t + 1])
            patient_id_map.append(patient_id)
            
            is_terminal = (t == len(states) - 2)
            reward = compute_outcome_reward(
                states[t], states[t + 1], actions_continuous[t],
                is_terminal, mortality, state_features
            )
            
            all_rewards.append(reward)
            all_dones.append(1.0 if is_terminal else 0.0)
    
    # Convert and normalize
    all_states = np.array(all_states, dtype=np.float32)
    all_actions_continuous = np.array(all_actions_continuous, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=np.float32)
    patient_id_map = np.array(patient_id_map)
    
    scaler = StandardScaler()
    all_states = scaler.fit_transform(all_states)
    all_next_states = scaler.transform(all_next_states)
    
    print(f"Total transitions: {len(all_states)}")
    print(f"Reward range: [{all_rewards.min():.2f}, {all_rewards.max():.2f}]")
    print(f"Action ranges: VP1=[{all_actions_continuous[:, 0].min():.2f}, {all_actions_continuous[:, 0].max():.2f}], "
          f"VP2=[{all_actions_continuous[:, 1].min():.3f}, {all_actions_continuous[:, 1].max():.3f}]")
    
    return {
        'states': all_states,
        'actions': all_actions_continuous,  # Continuous representation of discretized actions
        'rewards': all_rewards,
        'next_states': all_next_states,
        'dones': all_dones,
        'patient_ids': patient_id_map,
        'scaler': scaler,
        'state_features': state_features
    }


def train_single_cql(
    data: Dict,
    alpha: float,
    vp2_bins: int,
    epochs: int = 100,
    batch_size: int = 128,
    save_dir: str = 'experiment'
) -> Dict:
    """Train a single CQL agent with specific alpha and VP2 bins"""
    
    print(f"\n{'='*70}")
    print(f"Training CQL with alpha={alpha}, VP2_bins={vp2_bins}")
    print(f"{'='*70}")
    
    # Split data
    unique_patients = np.unique(data['patient_ids'])
    n_patients = len(unique_patients)
    
    # 70/15/15 split
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=0.15, random_state=42
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.15/0.85, random_state=42
    )
    
    train_idx = np.isin(data['patient_ids'], train_patients)
    val_idx = np.isin(data['patient_ids'], val_patients)
    test_idx = np.isin(data['patient_ids'], test_patients)
    
    print(f"Patient splits: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")
    
    # Initialize agent
    agent = DualDiscreteCQL(
        state_dim=len(data['state_features']),
        vp2_bins=vp2_bins,
        alpha=alpha,
        gamma=0.95,
        lr=0.001,
        tau=0.8,
        grad_clip=1.0
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(data['states'][train_idx]),
        torch.FloatTensor(data['actions'][train_idx]),  # Continuous actions
        torch.FloatTensor(data['rewards'][train_idx]),
        torch.FloatTensor(data['next_states'][train_idx]),
        torch.FloatTensor(data['dones'][train_idx])
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(data['states'][val_idx]),
        torch.FloatTensor(data['actions'][val_idx]),  # Continuous actions
        torch.FloatTensor(data['rewards'][val_idx]),
        torch.FloatTensor(data['next_states'][val_idx]),
        torch.FloatTensor(data['dones'][val_idx])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training metrics
    training_history = {
        'train_q_loss': [],
        'train_cql_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch+1}/{epochs}")
        # Training
        agent.q1.train()
        agent.q2.train()
        
        train_metrics = {'q1_loss': 0, 'cql1_loss': 0, 'total_q1_loss': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(train_loader)}...")
            states, actions, rewards, next_states, dones = [b.to(agent.device) for b in batch]
            metrics = agent.update(states, actions, rewards, next_states, dones)
            train_metrics['q1_loss'] += metrics['q1_loss']
            train_metrics['cql1_loss'] += metrics['cql1_loss']
            train_metrics['total_q1_loss'] += metrics['total_q1_loss']
        
        train_metrics['q1_loss'] /= len(train_loader)
        train_metrics['cql1_loss'] /= len(train_loader)
        train_metrics['total_q1_loss'] /= len(train_loader)
        
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
        
        # Store metrics
        training_history['train_q_loss'].append(train_metrics['q1_loss'])
        training_history['train_cql_loss'].append(train_metrics['cql1_loss'])
        training_history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = best_val_loss
            
            model_name = f'cql_discrete_alpha{alpha}_bins{vp2_bins}_best.pt'
            torch.save({
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'scaler': data['scaler'],
                'state_features': data['state_features'],
                'alpha': alpha,
                'vp2_bins': vp2_bins,
                'epoch': epoch
            }, os.path.join(save_dir, model_name))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Q-Loss={train_metrics['q1_loss']:.4f}, "
                  f"CQL-Loss={train_metrics['cql1_loss']:.4f}, "
                  f"Total-Loss={train_metrics['total_q1_loss']:.4f}, "
                  f"Val-Loss={val_loss:.4f}")
    
    # Save final model
    model_name = f'cql_discrete_alpha{alpha}_bins{vp2_bins}_final.pt'
    torch.save({
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'scaler': data['scaler'],
        'state_features': data['state_features'],
        'alpha': alpha,
        'vp2_bins': vp2_bins,
        'training_history': training_history
    }, os.path.join(save_dir, model_name))
    
    print(f"✅ Training completed for alpha={alpha}, bins={vp2_bins}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    
    return training_history


def main():
    """Run multiple CQL training experiments with different configurations"""
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiment/discrete_cql_{timestamp}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Configuration
    alpha_values = [0.001]  # Test with single alpha value
    vp2_bins_options = [5]  # Test with 5 bins only
    
    # Run experiments
    all_results = {}
    
    for vp2_bins in vp2_bins_options:
        print(f"\n{'='*80}")
        print(f"Starting experiments with VP2_bins={vp2_bins}")
        print(f"{'='*80}")
        
        # Load and prepare data once for each bin configuration
        data = prepare_data('sample_data_oviss.csv', vp2_bins)
        
        for alpha in alpha_values:
            exp_name = f'alpha{alpha}_bins{vp2_bins}'
            exp_dir = os.path.join(experiment_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Train model
            history = train_single_cql(
                data=data,
                alpha=alpha,
                vp2_bins=vp2_bins,
                epochs=100,
                batch_size=128,
                save_dir=exp_dir
            )
            
            all_results[exp_name] = {
                'alpha': alpha,
                'vp2_bins': vp2_bins,
                'best_val_loss': history['best_val_loss'],
                'final_train_q_loss': history['train_q_loss'][-1],
                'final_train_cql_loss': history['train_cql_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            }
    
    # Save summary results
    summary_path = os.path.join(experiment_dir, 'summary_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
        print(f"  Final Train Q-Loss: {results['final_train_q_loss']:.4f}")
        print(f"  Final Train CQL-Loss: {results['final_train_cql_loss']:.4f}")
        print(f"  Final Val Loss: {results['final_val_loss']:.4f}")
    
    # Find best configuration
    best_exp = min(all_results.items(), key=lambda x: x[1]['best_val_loss'])
    print(f"\n🏆 Best Configuration: {best_exp[0]}")
    print(f"   Best Val Loss: {best_exp[1]['best_val_loss']:.4f}")
    
    print(f"\n✅ All experiments completed!")
    print(f"Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()