"""
Off-Policy Evaluation for Block Discrete CQL Models using Q-value Softmax
===========================================================================

Q-value softmax-based importance sampling evaluation for block discrete CQL models.
Uses Q-values directly to compute action probabilities instead of KDE.
VP1: Binary (0 or 1)
VP2: Discretized into bins (0 to 0.5 mcg/kg/min)
Total actions: 2 * vp2_bins (e.g., 10 actions for vp2_bins=5)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the same data pipeline used for training
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2


# ============================================================================
# Q-Network Architectures
# ============================================================================

class DualBlockDiscreteQNetwork(nn.Module):
    """
    Q-network for block discrete actions using Q(s,a) -> R architecture
    Takes state and discrete action index as input, outputs single Q-value
    VP1: 2 actions (binary)
    VP2: vp2_bins actions (discretized continuous)
    Total: 2 * vp2_bins possible actions
    """
    
    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins  # VP1 (2) x VP2 (bins)
        
        # Network takes state + one-hot encoded action
        input_dim = state_dim + self.total_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)  # Output single Q-value
        
    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action_idx: [batch_size] - discrete action indices (0 to total_actions-1)
        Returns:
            q_value: [batch_size, 1] - Q-value for each (state, action) pair
        """
        # Convert action indices to one-hot encoding
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        
        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=-1)
        
        # Forward through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# CQL Agent Base Class
# ============================================================================

class CQLAgent:
    """Base class for CQL agents"""
    
    def __init__(self, state_dim: int, vp2_bins: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.device = device
        
        # Initialize Q-networks for block discrete action space
        self.q1 = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(device)
        self.q2 = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(device)
        
    def load(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        print(f"  Model loaded from {path}")
        
    def get_q_values_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions using double Q-learning"""
        with torch.no_grad():
            batch_size = states.shape[0]
            num_actions = self.q1.total_actions
            
            # Collect Q-values for all actions
            q1_values_list = []
            q2_values_list = []
            
            for action_idx in range(num_actions):
                action_tensor = torch.full((batch_size,), action_idx, dtype=torch.long).to(self.device)
                q1_val = self.q1(states, action_tensor)
                q2_val = self.q2(states, action_tensor)
                q1_values_list.append(q1_val.squeeze(-1))
                q2_values_list.append(q2_val.squeeze(-1))
            
            # Stack and take minimum
            q1_values = torch.stack(q1_values_list, dim=1)
            q2_values = torch.stack(q2_values_list, dim=1)
            return torch.min(q1_values, q2_values)


# ============================================================================
# Block Discrete CQL Agent
# ============================================================================

class BlockDiscreteCQLAgent(CQLAgent):
    """Block discrete action CQL agent (VP1 binary + VP2 discrete)"""
    
    def __init__(self, state_dim: int, vp2_bins: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(state_dim, vp2_bins=vp2_bins, device=device)
        
    def select_action(self, state: np.ndarray) -> int:
        """Select block discrete action index using Q-values"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get Q-values for all actions (2 * vp2_bins total actions)
            q_values = self.get_q_values_all_actions(state_tensor)
            
            # Select action with highest Q-value
            action_idx = torch.argmax(q_values, dim=1).item()
            
            return action_idx
    
    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """Select actions for a batch of states (vectorized)"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Get Q-values for all actions
            q_values = self.get_q_values_all_actions(states_tensor)
            
            # Select actions with highest Q-values
            action_indices = torch.argmax(q_values, dim=1).cpu().numpy()
            
            return action_indices
    
    def get_action_probabilities(self, states: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Compute action probabilities using Q-value softmax.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            temperature: Softmax temperature (lower = more deterministic)
        
        Returns:
            probs: Action probabilities [batch_size, num_actions]
        """
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Get Q-values for all actions
            q_values = self.get_q_values_all_actions(states_tensor)  # [batch_size, num_actions]
            
            # Apply softmax with temperature
            probs = torch.softmax(q_values / temperature, dim=1)
            
            return probs.cpu().numpy()
    
    def get_action_log_probabilities(self, states: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get log probabilities for numerical stability"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.get_q_values_all_actions(states_tensor)
            log_probs = torch.log_softmax(q_values / temperature, dim=1)
            return log_probs.cpu().numpy()


# ============================================================================
# Data Wrapper for IntegratedDataPipelineV2
# ============================================================================

class DataLoader:
    """Wrapper around IntegratedDataPipelineV2 for OPE"""
    
    def __init__(self, model_type: str = 'dual', vp2_bins: int = 5):
        """Initialize using the same pipeline as training"""
        self.pipeline = IntegratedDataPipelineV2(model_type=model_type, random_seed=42)
        self.vp2_bins = vp2_bins
        
        # Define VP2 bin edges (0 to 0.5 mcg/kg/min)
        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
        self.load_data()
        
    def load_data(self):
        """Load and preprocess data using IntegratedDataPipelineV2"""
        print("  Using IntegratedDataPipelineV2 with random_seed=42...")
        
        # Prepare data splits - this will match the training splits exactly
        train_data, val_data, test_data = self.pipeline.prepare_data()
        
        # Store the splits separately for proper evaluation
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Extract all data for OPE (we'll combine all splits)
        all_states = []
        all_actions = []
        all_rewards = []
        all_patient_ids = []
        all_splits = []  # Track which split each sample came from
        
        # Process each split
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if split_data is not None:
                states = split_data['states']
                actions = split_data['actions']  # This is [n_samples, 2] for dual model
                rewards = split_data['rewards']
                patient_ids = split_data.get('patient_ids', np.arange(len(states)))
                
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_patient_ids.append(patient_ids)
                all_splits.extend([split_name] * len(states))
        
        # Concatenate all data
        self.states = np.vstack(all_states)
        self.actions_raw = np.concatenate(all_actions)  # Keep raw format
        
        # Handle different model types
        if self.pipeline.model_type == 'binary':
            # Binary model: actions are already 1D binary (0 or 1)
            self.actions = self.actions_raw.astype(int)
        else:
            # Dual model: actions are 2D [VP1, VP2]
            # Convert actions to combined index format for block discrete
            # actions_raw[:, 0] is VP1 (binary)
            # actions_raw[:, 1] is VP2 (continuous, needs discretization)
            vp1_actions = self.actions_raw[:, 0].astype(int)  # Already binary
            
            # Discretize VP2 into bins
            vp2_continuous = self.actions_raw[:, 1]
            vp2_actions = np.digitize(vp2_continuous, self.vp2_bin_edges[1:])  # bins from 0 to vp2_bins-1
            vp2_actions = np.clip(vp2_actions, 0, self.vp2_bins - 1)
            
            # Combine into single action index: action = vp1 * vp2_bins + vp2
            self.actions = vp1_actions * self.vp2_bins + vp2_actions
        
        self.rewards = np.concatenate(all_rewards)
        self.patient_ids = np.concatenate(all_patient_ids)
        self.splits = np.array(all_splits)
        
        # Combine patient groups from all splits
        self.patient_groups = {}
        offset = 0
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if split_data is not None and 'patient_groups' in split_data:
                for patient_id, (start_idx, end_idx) in split_data['patient_groups'].items():
                    # Adjust indices to account for concatenation
                    self.patient_groups[patient_id] = (start_idx + offset, end_idx + offset)
                offset += len(split_data['states'])
        
        # States are already scaled by the pipeline
        self.states_scaled = self.states
        self.scaler = self.pipeline.scaler
        
        # Get unique patients
        unique_patients = len(np.unique(self.patient_ids[self.patient_ids > 0]))
        
        print(f"    Loaded {len(self.states)} transitions")
        print(f"    State dimension: {self.states.shape[1]}")
        print(f"    Train: {(self.splits == 'train').sum()}, Val: {(self.splits == 'val').sum()}, Test: {(self.splits == 'test').sum()}")
        
    def get_train_test_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get masks for train and test data based on pipeline splits"""
        # Use train+val for training KDE, test for evaluation
        # This ensures we evaluate on the exact same test set as the model was evaluated on
        train_mask = np.logical_or(self.splits == 'train', self.splits == 'val')
        test_mask = self.splits == 'test'
        
        return train_mask, test_mask


# ============================================================================
# Binary KDE Importance Sampling (for VP1 or simple binary actions)
# ============================================================================

class BinaryKDEImportanceSampler:
    """KDE-based importance sampling for binary actions"""
    
    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth
        
    def fit_policy(self, states: np.ndarray, actions: np.ndarray) -> Dict:
        """
        Fit KDE for a binary action policy
        
        Args:
            states: State observations [n_samples, state_dim]
            actions: Binary actions [n_samples] with values 0 or 1
            
        Returns:
            Dictionary containing:
                - 'prior': P(a=1) marginal probability
                - 'n_a0', 'n_a1': Number of samples for each action
                - 'kde_a0', 'kde_a1': Fitted KDE models for P(s|a)
        """
        # Separate states by action
        states_a0 = states[actions == 0]
        states_a1 = states[actions == 1]
        
        policy_kde = {
            'prior': actions.mean(),  # P(a=1)
            'n_a0': len(states_a0),
            'n_a1': len(states_a1)
        }
        
        # Fit KDEs for P(s|a=0) and P(s|a=1)
        if len(states_a0) > 0:
            policy_kde['kde_a0'] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            policy_kde['kde_a0'].fit(states_a0)
            
        if len(states_a1) > 0:
            policy_kde['kde_a1'] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            policy_kde['kde_a1'].fit(states_a1)
            
        return policy_kde
    
    def compute_action_prob(self, states: np.ndarray, policy_kde: Dict) -> np.ndarray:
        """
        Compute P(a=1|s) using Bayes rule
        
        Bayes rule: P(a=1|s) = P(s|a=1)P(a=1) / P(s)
        where P(s) = P(s|a=0)P(a=0) + P(s|a=1)P(a=1)
        
        Args:
            states: States to compute probabilities for [n_samples, state_dim]
            policy_kde: Fitted KDE policy dictionary
            
        Returns:
            P(a=1|s) for each state [n_samples]
        """
        n = len(states)
        
        # Handle edge cases
        if policy_kde['n_a0'] == 0:
            return np.ones(n) * policy_kde['prior']
        if policy_kde['n_a1'] == 0:
            return np.zeros(n)
            
        # Process in batches for efficiency with large datasets
        batch_size = 2000
        prob_a1 = np.zeros(n)
        
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            batch_states = states[i:end_idx]
            
            # Log probabilities P(s|a)
            log_p_s_a0 = policy_kde['kde_a0'].score_samples(batch_states)
            log_p_s_a1 = policy_kde['kde_a1'].score_samples(batch_states)
            
            # Log priors P(a)
            log_p_a0 = np.log(1 - policy_kde['prior'] + 1e-10)
            log_p_a1 = np.log(policy_kde['prior'] + 1e-10)
            
            # Log joint probabilities P(s,a) = P(s|a)P(a)
            log_joint_a0 = log_p_s_a0 + log_p_a0
            log_joint_a1 = log_p_s_a1 + log_p_a1
            
            # Normalize using log-sum-exp for numerical stability
            log_normalizer = np.logaddexp(log_joint_a0, log_joint_a1)
            prob_a1[i:end_idx] = np.exp(log_joint_a1 - log_normalizer)
        
        return np.clip(prob_a1, 1e-6, 1-1e-6)
    
    def compute_importance_weights(self, states: np.ndarray, actions: np.ndarray,
                                  behavioral_kde: Dict, target_kde: Dict) -> np.ndarray:
        """
        Compute importance weights w(s,a) = π_target(a|s) / π_behavioral(a|s)
        
        Args:
            states: States [n_samples, state_dim]
            actions: Binary actions taken [n_samples]
            behavioral_kde: KDE for behavioral policy
            target_kde: KDE for target policy
            
        Returns:
            Importance weights [n_samples]
        """
        # Get P(a=1|s) for both policies
        prob_behavioral = self.compute_action_prob(states, behavioral_kde)
        prob_target = self.compute_action_prob(states, target_kde)
        
        # Compute weights based on actual actions
        weights = np.zeros(len(states))
        
        # For a=1: w = P_target(a=1|s) / P_behavioral(a=1|s)
        mask_a1 = (actions == 1)
        weights[mask_a1] = prob_target[mask_a1] / (prob_behavioral[mask_a1] + 1e-10)
        
        # For a=0: w = P_target(a=0|s) / P_behavioral(a=0|s)
        #           = (1 - P_target(a=1|s)) / (1 - P_behavioral(a=1|s))
        mask_a0 = (actions == 0)
        weights[mask_a0] = (1 - prob_target[mask_a0]) / (1 - prob_behavioral[mask_a0] + 1e-10)
        
        return np.clip(weights, 0, 20)


# ============================================================================
# Block Discrete KDE Importance Sampling (for VP1 + VP2 joint actions)
# ============================================================================

class BlockDiscreteKDEImportanceSampler:
    """
    KDE-based importance sampling for block discrete actions (VP1 x VP2)
    Uses factorization: P(a1,a2|s) = P(a2|a1,s) * P(a1|s)
    where a1 is VP1 (binary) and a2 is VP2 (discrete)
    """
    
    def __init__(self, bandwidth: float = 0.1, vp2_bins: int = 5):
        self.bandwidth = bandwidth
        self.vp2_bins = vp2_bins
        self.num_actions = 2 * vp2_bins  # Total combined actions
        
    def decode_action(self, action_idx: int) -> Tuple[int, int]:
        """
        Decode combined action index into VP1 and VP2 actions
        Action encoding: action_idx = vp1 * vp2_bins + vp2
        
        Args:
            action_idx: Combined action index (0 to num_actions-1)
            
        Returns:
            (vp1, vp2) tuple where vp1 in {0,1} and vp2 in {0,...,vp2_bins-1}
        """
        vp1 = action_idx // self.vp2_bins
        vp2 = action_idx % self.vp2_bins
        return vp1, vp2
    
    def encode_action(self, vp1: int, vp2: int) -> int:
        """
        Encode VP1 and VP2 into combined action index
        
        Args:
            vp1: Binary action for VP1 (0 or 1)
            vp2: Discrete action for VP2 (0 to vp2_bins-1)
            
        Returns:
            Combined action index
        """
        return vp1 * self.vp2_bins + vp2
    
    def fit_policy(self, states: np.ndarray, actions: np.ndarray) -> Dict:
        """
        Fit KDE for a policy with factorized conditional probabilities
        P(a1,a2|s) = P(a2|a1,s) * P(a1|s)
        
        Args:
            states: State observations [n_samples, state_dim]
            actions: Combined action indices [n_samples]
        
        Returns:
            Dictionary containing KDEs for P(a1|s) and P(a2|a1,s)
        """
        n_samples = len(actions)
        
        # Decode actions into VP1 and VP2
        vp1_actions = np.zeros(n_samples, dtype=int)
        vp2_actions = np.zeros(n_samples, dtype=int)
        
        for i, action_idx in enumerate(actions):
            vp1_actions[i], vp2_actions[i] = self.decode_action(int(action_idx))
        
        policy_kde = {
            'vp2_bins': self.vp2_bins,
            'total_samples': n_samples
        }
        
        # 1. Fit marginal P(a1|s) for VP1
        states_vp1_0 = states[vp1_actions == 0]
        states_vp1_1 = states[vp1_actions == 1]
        
        policy_kde['vp1_prior'] = vp1_actions.mean()  # P(vp1=1)
        policy_kde['n_vp1_0'] = len(states_vp1_0)
        policy_kde['n_vp1_1'] = len(states_vp1_1)
        
        if len(states_vp1_0) > 0:
            policy_kde['kde_vp1_0'] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            policy_kde['kde_vp1_0'].fit(states_vp1_0)
            
        if len(states_vp1_1) > 0:
            policy_kde['kde_vp1_1'] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            policy_kde['kde_vp1_1'].fit(states_vp1_1)
        
        # 2. Fit conditional P(a2|a1,s) for VP2 given VP1
        for vp1_val in [0, 1]:
            mask_vp1 = (vp1_actions == vp1_val)
            states_given_vp1 = states[mask_vp1]
            vp2_given_vp1 = vp2_actions[mask_vp1]
            
            if len(states_given_vp1) > 0:
                # Store conditional priors P(vp2|vp1)
                for vp2_val in range(self.vp2_bins):
                    mask_vp2 = (vp2_given_vp1 == vp2_val)
                    n_samples_vp2 = mask_vp2.sum()
                    
                    key_prefix = f'vp1_{vp1_val}_vp2_{vp2_val}'
                    policy_kde[f'{key_prefix}_prior'] = n_samples_vp2 / len(vp2_given_vp1)
                    policy_kde[f'{key_prefix}_n'] = n_samples_vp2
                    
                    if n_samples_vp2 > 0:
                        states_vp2 = states_given_vp1[mask_vp2]
                        kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
                        kde.fit(states_vp2)
                        policy_kde[f'{key_prefix}_kde'] = kde
        
        return policy_kde
    
    def compute_vp1_prob(self, states: np.ndarray, policy_kde: Dict) -> np.ndarray:
        """
        Compute P(vp1=1|s) using Bayes rule
        
        Args:
            states: States to compute probabilities for [n_samples, state_dim]
            policy_kde: Fitted KDE policy dictionary
            
        Returns:
            P(vp1=1|s) for each state [n_samples]
        """
        n = len(states)
        
        if policy_kde['n_vp1_0'] == 0:
            return np.ones(n) * policy_kde['vp1_prior']
        if policy_kde['n_vp1_1'] == 0:
            return np.zeros(n)
        
        # Process in batches
        batch_size = 2000
        prob_vp1_1 = np.zeros(n)
        
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            batch_states = states[i:end_idx]
            
            # Log probabilities P(s|vp1)
            log_p_s_vp1_0 = policy_kde['kde_vp1_0'].score_samples(batch_states)
            log_p_s_vp1_1 = policy_kde['kde_vp1_1'].score_samples(batch_states)
            
            # Priors P(vp1)
            log_p_vp1_0 = np.log(1 - policy_kde['vp1_prior'] + 1e-10)
            log_p_vp1_1 = np.log(policy_kde['vp1_prior'] + 1e-10)
            
            # Joint P(s,vp1)
            log_joint_vp1_0 = log_p_s_vp1_0 + log_p_vp1_0
            log_joint_vp1_1 = log_p_s_vp1_1 + log_p_vp1_1
            
            # Normalize
            log_normalizer = np.logaddexp(log_joint_vp1_0, log_joint_vp1_1)
            prob_vp1_1[i:end_idx] = np.exp(log_joint_vp1_1 - log_normalizer)
        
        return np.clip(prob_vp1_1, 1e-6, 1-1e-6)
    
    def compute_vp2_prob_given_vp1(self, states: np.ndarray, vp1: int, 
                                   policy_kde: Dict) -> np.ndarray:
        """
        Compute P(vp2|vp1,s) for all VP2 values using Bayes rule
        
        Args:
            states: States to compute probabilities for [n_samples, state_dim]
            vp1: Conditioning VP1 value (0 or 1)
            policy_kde: Fitted KDE policy dictionary
            
        Returns:
            P(vp2|vp1,s) for each VP2 value [n_states, vp2_bins]
        """
        n_states = len(states)
        vp2_probs = np.zeros((n_states, self.vp2_bins))
        
        # Process in batches
        batch_size = 2000
        
        for i in range(0, n_states, batch_size):
            end_idx = min(i + batch_size, n_states)
            batch_states = states[i:end_idx]
            batch_size_actual = end_idx - i
            
            # Compute log probabilities for each VP2 value
            log_probs = np.full((batch_size_actual, self.vp2_bins), -np.inf)
            
            for vp2_val in range(self.vp2_bins):
                key_prefix = f'vp1_{vp1}_vp2_{vp2_val}'
                
                if f'{key_prefix}_kde' in policy_kde and policy_kde[f'{key_prefix}_n'] > 0:
                    # Log likelihood P(s|vp1,vp2)
                    log_likelihood = policy_kde[f'{key_prefix}_kde'].score_samples(batch_states)
                    
                    # Log conditional prior P(vp2|vp1)
                    log_prior = np.log(policy_kde[f'{key_prefix}_prior'] + 1e-10)
                    
                    # Log joint P(s,vp2|vp1)
                    log_probs[:, vp2_val] = log_likelihood + log_prior
            
            # Normalize to get P(vp2|vp1,s)
            log_normalizer = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
            vp2_probs[i:end_idx] = np.exp(log_probs - log_normalizer)
        
        return np.clip(vp2_probs, 1e-8, 1.0)
    
    def compute_joint_action_probs(self, states: np.ndarray, policy_kde: Dict) -> np.ndarray:
        """
        Compute P(a1,a2|s) = P(a2|a1,s) * P(a1|s) for all action combinations
        
        Args:
            states: States to compute probabilities for [n_samples, state_dim]
            policy_kde: Fitted KDE policy dictionary
            
        Returns:
            Joint action probabilities [n_states, num_actions]
        """
        n_states = len(states)
        joint_probs = np.zeros((n_states, self.num_actions))
        
        # Get P(vp1|s)
        prob_vp1_1 = self.compute_vp1_prob(states, policy_kde)
        prob_vp1_0 = 1 - prob_vp1_1
        
        # For VP1=0: P(vp1=0,vp2|s) = P(vp2|vp1=0,s) * P(vp1=0|s)
        vp2_probs_given_vp1_0 = self.compute_vp2_prob_given_vp1(states, 0, policy_kde)
        for vp2_val in range(self.vp2_bins):
            action_idx = self.encode_action(0, vp2_val)
            joint_probs[:, action_idx] = prob_vp1_0 * vp2_probs_given_vp1_0[:, vp2_val]
        
        # For VP1=1: P(vp1=1,vp2|s) = P(vp2|vp1=1,s) * P(vp1=1|s)
        vp2_probs_given_vp1_1 = self.compute_vp2_prob_given_vp1(states, 1, policy_kde)
        for vp2_val in range(self.vp2_bins):
            action_idx = self.encode_action(1, vp2_val)
            joint_probs[:, action_idx] = prob_vp1_1 * vp2_probs_given_vp1_1[:, vp2_val]
        
        return joint_probs
    
    def compute_importance_weights(self, states: np.ndarray, actions: np.ndarray,
                                  behavioral_kde: Dict, target_kde: Dict) -> np.ndarray:
        """
        Compute importance weights w(s,a) = π_target(a|s) / π_behavioral(a|s)
        Using factorized probabilities P(a1,a2|s) = P(a2|a1,s) * P(a1|s)
        
        Args:
            states: States [n_samples, state_dim]
            actions: Combined action indices [n_samples]
            behavioral_kde: KDE for behavioral policy
            target_kde: KDE for target policy
            
        Returns:
            Importance weights [n_samples]
        """
        # Get joint action probabilities for both policies
        prob_behavioral = self.compute_joint_action_probs(states, behavioral_kde)
        prob_target = self.compute_joint_action_probs(states, target_kde)
        
        # Extract probabilities for actual actions taken
        n_samples = len(actions)
        weights = np.zeros(n_samples)
        
        for i in range(n_samples):
            action_idx = int(actions[i])
            weights[i] = prob_target[i, action_idx] / (prob_behavioral[i, action_idx] + 1e-10)
        
        return np.clip(weights, 0, 20)


# ============================================================================
# Q-Softmax Importance Sampling (Simple backup version)
# ============================================================================

class QSoftmaxImportanceSampler:
    """Simple Q-value softmax-based importance sampling"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def get_action_probs(self, states: np.ndarray, agent: BlockDiscreteCQLAgent, 
                        temperature: float = None) -> np.ndarray:
        """Get action probabilities from Q-values using softmax"""
        if temperature is None:
            temperature = self.temperature
            
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(agent.device)
            batch_size = states_tensor.shape[0]
            num_actions = agent.q1.total_actions
            
            # Collect Q-values for all actions
            q_values_list = []
            for action_idx in range(num_actions):
                action_tensor = torch.full((batch_size,), action_idx, dtype=torch.long).to(agent.device)
                q1 = agent.q1(states_tensor, action_tensor)
                q2 = agent.q2(states_tensor, action_tensor)
                q_values_list.append(torch.min(q1, q2).squeeze(-1))
            
            # Stack and apply softmax
            q_values = torch.stack(q_values_list, dim=1)
            probs = torch.softmax(q_values / temperature, dim=1)
            
            return probs.cpu().numpy()
    
    def compute_importance_weights(self, states: np.ndarray, actions: np.ndarray,
                                  target_agent: BlockDiscreteCQLAgent,
                                  behavioral_agent: BlockDiscreteCQLAgent = None) -> np.ndarray:
        """Compute importance weights w(s,a) = π_target(a|s) / π_behavioral(a|s)"""
        
        # Get probabilities
        prob_target = self.get_action_probs(states, target_agent)
        
        if behavioral_agent is not None:
            prob_behavioral = self.get_action_probs(states, behavioral_agent)
        else:
            # Assume uniform behavioral policy
            prob_behavioral = np.ones_like(prob_target) / prob_target.shape[1]
        
        # Compute weights for actual actions
        weights = np.zeros(len(actions))
        for i, action in enumerate(actions):
            weights[i] = prob_target[i, int(action)] / (prob_behavioral[i, int(action)] + 1e-10)
        
        return np.clip(weights, 0, 20)


# ============================================================================
# OPE Evaluator
# ============================================================================

class OPEEvaluator:
    """Off-Policy Evaluation using multiple importance sampling methods"""
    
    def __init__(self, method: str = 'block_kde', kde_bandwidth: float = 0.1, 
                 vp2_bins: int = 5, temperature: float = 1.0):
        """
        Args:
            method: 'binary_kde', 'block_kde', or 'q_softmax'
            kde_bandwidth: Bandwidth for KDE methods
            vp2_bins: Number of VP2 bins for block discrete actions
            temperature: Softmax temperature for Q-softmax method
        """
        self.method = method
        self.vp2_bins = vp2_bins
        
        if method == 'binary_kde':
            self.sampler = BinaryKDEImportanceSampler(bandwidth=kde_bandwidth)
        elif method == 'block_kde':
            self.sampler = BlockDiscreteKDEImportanceSampler(bandwidth=kde_bandwidth, vp2_bins=vp2_bins)
        elif method == 'q_softmax':
            self.sampler = QSoftmaxImportanceSampler(temperature=temperature)
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def evaluate(self, data: DataLoader, agent: BlockDiscreteCQLAgent,
                train_mask: np.ndarray, test_mask: np.ndarray) -> Dict:
        """
        Evaluate agent using importance sampling
        
        Returns:
            Dictionary with evaluation metrics
        """
        
        # Generate learned policy actions
        print("\n  Generating learned policy actions...")
        batch_size = 10000
        learned_actions = np.zeros(len(data.states), dtype=int)
        
        for i in range(0, len(data.states), batch_size):
            end_idx = min(i + batch_size, len(data.states))
            batch_states = data.states_scaled[i:end_idx]
            
            # Get actions for the batch (vectorized)
            batch_actions = agent.select_actions_batch(batch_states)
            learned_actions[i:end_idx] = batch_actions
            
            if (i // batch_size) % 2 == 0:
                print(f"    Processed {end_idx}/{len(data.states)} samples...")
        
        # Compute importance weights based on method
        print(f"\n  Computing importance weights using {self.method}...")
        
        if self.method in ['binary_kde', 'block_kde']:
            # KDE-based methods
            print("  Fitting KDE models...")
            print(f"    Training samples: {train_mask.sum()}")
            
            # Subsample training data for faster KDE fitting
            max_kde_samples = 10000
            train_indices = np.where(train_mask)[0]
            if len(train_indices) > max_kde_samples:
                np.random.seed(42)
                train_indices = np.random.choice(train_indices, max_kde_samples, replace=False)
                print(f"    Subsampling to {max_kde_samples} samples for KDE fitting")
            
            # Fit KDEs
            behavioral_kde = self.sampler.fit_policy(
                data.states_scaled[train_indices],
                data.actions[train_indices]
            )
            print("    Behavioral KDE fitted")
            
            target_kde = self.sampler.fit_policy(
                data.states_scaled[train_indices],
                learned_actions[train_indices]
            )
            print("    Target KDE fitted")
            
            # Compute importance weights on test set
            weights = self.sampler.compute_importance_weights(
                data.states_scaled[test_mask],
                data.actions[test_mask],
                behavioral_kde,
                target_kde
            )
            
        elif self.method == 'q_softmax':
            # Q-softmax method - no KDE fitting needed
            print("  Using Q-values directly (no KDE fitting required)")
            
            # For Q-softmax, we use the learned agent as target
            # and can either use uniform or another agent as behavioral
            weights = self.sampler.compute_importance_weights(
                data.states_scaled[test_mask],
                data.actions[test_mask],
                target_agent=agent,
                behavioral_agent=None  # Use uniform behavioral policy
            )
        
        print(f"    Computed weights for {len(weights)} test samples")
        
        # Compute estimates
        rewards_test = data.rewards[test_mask]
        
        # Importance Sampling
        is_estimate = np.mean(weights * rewards_test)
        
        # Weighted Importance Sampling
        wis_estimate = np.sum(weights * rewards_test) / np.sum(weights)
        
        # Effective Sample Size
        ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
        
        # Confidence intervals via bootstrap
        n_bootstrap = 100
        is_bootstrap = []
        wis_bootstrap = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(rewards_test), len(rewards_test), replace=True)
            w_b = weights[idx]
            r_b = rewards_test[idx]
            
            is_bootstrap.append(np.mean(w_b * r_b))
            wis_bootstrap.append(np.sum(w_b * r_b) / np.sum(w_b))
        
        results = {
            'behavioral_reward': rewards_test.mean(),
            'is_estimate': is_estimate,
            'wis_estimate': wis_estimate,
            'is_ci': (np.percentile(is_bootstrap, 2.5), np.percentile(is_bootstrap, 97.5)),
            'wis_ci': (np.percentile(wis_bootstrap, 2.5), np.percentile(wis_bootstrap, 97.5)),
            'ess': ess,
            'ess_ratio': ess / len(rewards_test),
            'weights': weights,
            'weight_mean': weights.mean(),
            'weight_std': weights.std(),
            'weight_max': weights.max()
        }
        
        return results


# ============================================================================
# Visualization
# ============================================================================

def visualize_ope_results(results: Dict, save_path: Optional[str] = None):
    """Visualize OPE results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Weight distribution
    ax = axes[0]
    weights = results['weights']
    ax.hist(weights, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', label='w=1')
    ax.set_xlabel('Importance Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Weight Distribution')
    ax.legend()
    
    # Estimates comparison
    ax = axes[1]
    estimates = {
        'Behavioral': results['behavioral_reward'],
        'IS': results['is_estimate'],
        'WIS': results['wis_estimate']
    }
    
    x_pos = np.arange(len(estimates))
    values = list(estimates.values())
    colors = ['blue', 'orange', 'green']
    
    bars = ax.bar(x_pos, values, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(estimates.keys())
    ax.set_ylabel('Expected Reward')
    ax.set_title('OPE Estimates')
    
    # Add confidence intervals
    ax.errorbar(1, results['is_estimate'],
               yerr=[[results['is_estimate'] - results['is_ci'][0]],
                     [results['is_ci'][1] - results['is_estimate']]],
               fmt='none', color='black', capsize=5)
    ax.errorbar(2, results['wis_estimate'],
               yerr=[[results['wis_estimate'] - results['wis_ci'][0]],
                     [results['wis_ci'][1] - results['wis_estimate']]],
               fmt='none', color='black', capsize=5)
    
    # ESS visualization
    ax = axes[2]
    ax.text(0.5, 0.5, f"ESS: {results['ess']:.1f}\n({results['ess_ratio']*100:.1f}% of samples)",
           ha='center', va='center', fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    ax.set_title('Effective Sample Size')
    
    plt.suptitle('CQL Off-Policy Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Visualization saved to {save_path}")
    
    plt.show()


# ============================================================================
# Binary CQL Agent (for comparison)
# ============================================================================

class BinaryCQLAgent:
    """Binary action CQL agent (VP1 only)"""
    
    def __init__(self, state_dim: int, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = 1
        self.device = device
        
        # Initialize Q-networks for continuous actions
        self.q1 = ContinuousQNetwork(state_dim, 1).to(device)
        self.q2 = ContinuousQNetwork(state_dim, 1).to(device)
        
    def load(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        print(f"  Model loaded from {path}")
        
    def select_action(self, state: np.ndarray) -> float:
        """Select binary action using Q-values"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Evaluate Q-values for both actions
            q_a0 = self.q1(state_tensor, torch.zeros(1, 1).to(self.device))
            q_a1 = self.q1(state_tensor, torch.ones(1, 1).to(self.device))
            
            # Use Q2 as well for double Q-learning
            q2_a0 = self.q2(state_tensor, torch.zeros(1, 1).to(self.device))
            q2_a1 = self.q2(state_tensor, torch.ones(1, 1).to(self.device))
            
            # Take minimum for conservative estimate
            q_a0 = torch.min(q_a0, q2_a0).item()
            q_a1 = torch.min(q_a1, q2_a1).item()
            
            # Return action with higher Q-value
            return 1.0 if q_a1 > q_a0 else 0.0
    
    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """Select actions for a batch of states"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            batch_size = states_tensor.shape[0]
            
            # Evaluate Q-values for both actions
            zeros = torch.zeros(batch_size, 1).to(self.device)
            ones = torch.ones(batch_size, 1).to(self.device)
            
            q1_a0 = self.q1(states_tensor, zeros)
            q1_a1 = self.q1(states_tensor, ones)
            q2_a0 = self.q2(states_tensor, zeros)
            q2_a1 = self.q2(states_tensor, ones)
            
            # Take minimum for conservative estimate
            q_a0 = torch.min(q1_a0, q2_a0).squeeze()
            q_a1 = torch.min(q1_a1, q2_a1).squeeze()
            
            # Return actions with higher Q-values
            actions = (q_a1 > q_a0).float().cpu().numpy()
            
            return actions


class ContinuousQNetwork(nn.Module):
    """Standard Q-network for continuous actions: Q(s,a) -> R"""
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# Main Evaluation Functions
# ============================================================================

def evaluate_binary_cql(model_path: str, method: str = 'binary_kde',
                       temperature: float = 1.0) -> Dict:
    """
    Evaluate a binary CQL model using importance sampling
    
    Args:
        model_path: Path to saved model
        method: 'binary_kde' or 'q_softmax'
        temperature: Softmax temperature for q_softmax method
        
    Returns:
        Dictionary with evaluation results
    """
    
    print("=" * 60)
    print("BINARY CQL OFF-POLICY EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Method: {method}")
    
    # Load data - use 'binary' model type to get VP1 only
    print("\n1. Loading data...")
    data = DataLoader(model_type='binary', vp2_bins=1)  # vp2_bins not used for binary
    
    # Load model
    print("\n2. Loading CQL model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    agent = BinaryCQLAgent(state_dim=data.states.shape[1])
    agent.load(model_path)
    
    # Split data
    print("\n3. Splitting data...")
    train_mask, test_mask = data.get_train_test_masks()
    print(f"  Train+Val: {train_mask.sum()} samples")
    print(f"  Test: {test_mask.sum()} samples")
    
    # Evaluate
    print("\n4. Running OPE...")
    evaluator = OPEEvaluator(method=method, vp2_bins=1, temperature=temperature)
    
    # For binary model, we need to handle the evaluation differently
    # Generate learned policy actions
    print("\n  Generating learned policy actions...")
    batch_size = 10000
    learned_actions = np.zeros(len(data.states), dtype=int)
    
    for i in range(0, len(data.states), batch_size):
        end_idx = min(i + batch_size, len(data.states))
        batch_states = data.states_scaled[i:end_idx]
        batch_actions = agent.select_actions_batch(batch_states)
        learned_actions[i:end_idx] = batch_actions.astype(int)
        
        if (i // batch_size) % 2 == 0:
            print(f"    Processed {end_idx}/{len(data.states)} samples...")
    
    # Compute importance weights
    print(f"\n  Computing importance weights using {method}...")
    
    # Subsample training data for KDE
    max_kde_samples = 10000
    train_indices = np.where(train_mask)[0]
    if len(train_indices) > max_kde_samples:
        np.random.seed(42)
        train_indices = np.random.choice(train_indices, max_kde_samples, replace=False)
        print(f"    Subsampling to {max_kde_samples} samples for KDE fitting")
    
    # Use BinaryKDEImportanceSampler
    sampler = BinaryKDEImportanceSampler(bandwidth=0.1)
    
    # Fit KDEs
    behavioral_kde = sampler.fit_policy(
        data.states_scaled[train_indices],
        data.actions[train_indices]
    )
    print("    Behavioral KDE fitted")
    
    target_kde = sampler.fit_policy(
        data.states_scaled[train_indices],
        learned_actions[train_indices]
    )
    print("    Target KDE fitted")
    
    # Compute importance weights on test set
    weights = sampler.compute_importance_weights(
        data.states_scaled[test_mask],
        data.actions[test_mask],
        behavioral_kde,
        target_kde
    )
    
    print(f"    Computed weights for {len(weights)} test samples")
    
    # Compute estimates
    rewards_test = data.rewards[test_mask]
    
    # Importance Sampling
    is_estimate = np.mean(weights * rewards_test)
    
    # Weighted Importance Sampling
    wis_estimate = np.sum(weights * rewards_test) / np.sum(weights)
    
    # Effective Sample Size
    ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
    
    # Confidence intervals via bootstrap
    n_bootstrap = 100
    is_bootstrap = []
    wis_bootstrap = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(rewards_test), len(rewards_test), replace=True)
        w_b = weights[idx]
        r_b = rewards_test[idx]
        
        is_bootstrap.append(np.mean(w_b * r_b))
        wis_bootstrap.append(np.sum(w_b * r_b) / np.sum(w_b))
    
    results = {
        'behavioral_reward': rewards_test.mean(),
        'is_estimate': is_estimate,
        'wis_estimate': wis_estimate,
        'is_ci': (np.percentile(is_bootstrap, 2.5), np.percentile(is_bootstrap, 97.5)),
        'wis_ci': (np.percentile(wis_bootstrap, 2.5), np.percentile(wis_bootstrap, 97.5)),
        'ess': ess,
        'ess_ratio': ess / len(rewards_test),
        'weights': weights,
        'weight_mean': weights.mean(),
        'weight_std': weights.std(),
        'weight_max': weights.max()
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nBehavioral Policy:")
    print(f"  Average Reward: {results['behavioral_reward']:.4f}")
    
    print(f"\nLearned Policy Estimates:")
    print(f"  IS:  {results['is_estimate']:.4f} "
          f"(95% CI: [{results['is_ci'][0]:.4f}, {results['is_ci'][1]:.4f}])")
    print(f"  WIS: {results['wis_estimate']:.4f} "
          f"(95% CI: [{results['wis_ci'][0]:.4f}, {results['wis_ci'][1]:.4f}])")
    
    print(f"\nImportance Weights:")
    print(f"  Mean: {results['weight_mean']:.3f}")
    print(f"  Std:  {results['weight_std']:.3f}")
    print(f"  Max:  {results['weight_max']:.3f}")
    print(f"  ESS:  {results['ess']:.1f} ({results['ess_ratio']*100:.1f}%)")
    
    return results


def evaluate_block_discrete_cql(model_path: str, vp2_bins: int = 10, 
                                method: str = 'block_kde', 
                                temperature: float = 1.0) -> Dict:
    """
    Evaluate a block discrete CQL model using importance sampling
    
    Args:
        model_path: Path to saved model (e.g., 'experiment/block_discrete_cql_alpha0.0000_bins10_best.pt')
        vp2_bins: Number of VP2 bins (must match the model)
        method: 'block_kde', 'binary_kde', or 'q_softmax'
        temperature: Softmax temperature for q_softmax method
        
    Returns:
        Dictionary with evaluation results
    """
    
    print("=" * 60)
    print("BLOCK DISCRETE CQL OFF-POLICY EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"VP2 bins: {vp2_bins}")
    print(f"Method: {method}")
    
    # Load data
    print("\n1. Loading data...")
    data = DataLoader(model_type='dual', vp2_bins=vp2_bins)
    
    # Load model
    print("\n2. Loading CQL model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    agent = BlockDiscreteCQLAgent(state_dim=data.states.shape[1], vp2_bins=vp2_bins)
    agent.load(model_path)
    
    # Split data
    print("\n3. Splitting data...")
    train_mask, test_mask = data.get_train_test_masks()
    print(f"  Train+Val: {train_mask.sum()} samples")
    print(f"  Test: {test_mask.sum()} samples")
    
    # Evaluate
    print("\n4. Running OPE...")
    evaluator = OPEEvaluator(method=method, vp2_bins=vp2_bins, temperature=temperature)
    results = evaluator.evaluate(data, agent, train_mask, test_mask)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nBehavioral Policy:")
    print(f"  Average Reward: {results['behavioral_reward']:.4f}")
    
    print(f"\nLearned Policy Estimates:")
    print(f"  IS:  {results['is_estimate']:.4f} "
          f"(95% CI: [{results['is_ci'][0]:.4f}, {results['is_ci'][1]:.4f}])")
    print(f"  WIS: {results['wis_estimate']:.4f} "
          f"(95% CI: [{results['wis_ci'][0]:.4f}, {results['wis_ci'][1]:.4f}])")
    
    print(f"\nImportance Weights:")
    print(f"  Mean: {results['weight_mean']:.3f}")
    print(f"  Std:  {results['weight_std']:.3f}")
    print(f"  Max:  {results['weight_max']:.3f}")
    print(f"  ESS:  {results['ess']:.1f} ({results['ess_ratio']*100:.1f}%)")
    
    # Visualize
    visualize_ope_results(results, save_path='kde_demo/cql_ope_results.png')
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Block Discrete CQL with OPE')
    parser.add_argument('--model_path', type=str, 
                       default='experiment/block_discrete_cql_alpha0.0000_bins10_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vp2_bins', type=int, default=10,
                       help='Number of VP2 bins (must match model)')
    parser.add_argument('--method', type=str, default='block_kde',
                       choices=['binary_kde', 'block_kde', 'q_softmax'],
                       help='Importance sampling method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Softmax temperature for q_softmax method')
    parser.add_argument('--compare_methods', action='store_true',
                       help='Compare all three methods')
    
    args = parser.parse_args()
    
    if args.compare_methods:
        # Compare all three methods
        print("\n" + "="*80)
        print("COMPARING ALL IMPORTANCE SAMPLING METHODS")
        print("="*80)
        
        all_results = {}
        for method in ['block_kde', 'q_softmax']:
            print(f"\n\n>>> Method: {method.upper()}")
            results = evaluate_block_discrete_cql(
                args.model_path, 
                vp2_bins=args.vp2_bins, 
                method=method,
                temperature=args.temperature
            )
            all_results[method] = results
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Method':<15} {'Behavioral':<12} {'IS Estimate':<12} {'WIS Estimate':<12} {'ESS Ratio':<12}")
        print("-"*63)
        
        for method, res in all_results.items():
            print(f"{method:<15} {res['behavioral_reward']:.4f}      "
                  f"{res['is_estimate']:.4f}       {res['wis_estimate']:.4f}       "
                  f"{res['ess_ratio']*100:.1f}%")
    else:
        # Single method evaluation
        results = evaluate_block_discrete_cql(
            args.model_path,
            vp2_bins=args.vp2_bins,
            method=args.method,
            temperature=args.temperature
        )