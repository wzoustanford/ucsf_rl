"""
Off-Policy Evaluation for Block Discrete CQL Models
====================================================

KDE-based importance sampling evaluation for block discrete CQL models.
VP1: Binary (0 or 1)
VP2: Discretized into bins (0 to 0.5 mcg/kg/min)
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
        
    def get_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions using double Q-learning"""
        with torch.no_grad():
            q1_values = self.q1(states)
            q2_values = self.q2(states)
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
            q_values = self.get_q_values(state_tensor)
            
            # Select action with highest Q-value
            action_idx = torch.argmax(q_values, dim=1).item()
            
            return action_idx
    
    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """Select actions for a batch of states (vectorized)"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Get Q-values for all actions
            q_values = self.get_q_values(states_tensor)
            
            # Select actions with highest Q-values
            action_indices = torch.argmax(q_values, dim=1).cpu().numpy()
            
            return action_indices


# ============================================================================
# Data Wrapper for IntegratedDataPipelineV2
# ============================================================================

class DataLoader:
    """Wrapper around IntegratedDataPipelineV2 for OPE"""
    
    def __init__(self, model_type: str = 'binary'):
        """Initialize using the same pipeline as training"""
        self.pipeline = IntegratedDataPipelineV2(model_type=model_type, random_seed=42)
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
                actions = split_data['actions'] 
                rewards = split_data['rewards']
                patient_ids = split_data.get('patient_ids', np.arange(len(states)))
                
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_patient_ids.append(patient_ids)
                all_splits.extend([split_name] * len(states))
        
        # Concatenate all data
        self.states = np.vstack(all_states)
        self.actions = np.concatenate(all_actions)
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
# KDE Importance Sampling
# ============================================================================

class KDEImportanceSampler:
    """KDE-based importance sampling for binary actions"""
    
    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth
        
    def fit_policy(self, states: np.ndarray, actions: np.ndarray) -> Dict:
        """Fit KDE for a policy"""
        # Separate states by action
        states_a0 = states[actions == 0]
        states_a1 = states[actions == 1]
        
        policy_kde = {
            'prior': actions.mean(),  # P(a=1)
            'n_a0': len(states_a0),
            'n_a1': len(states_a1)
        }
        
        # Fit KDEs
        if len(states_a0) > 0:
            policy_kde['kde_a0'] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            policy_kde['kde_a0'].fit(states_a0)
            
        if len(states_a1) > 0:
            policy_kde['kde_a1'] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            policy_kde['kde_a1'].fit(states_a1)
            
        return policy_kde
    
    def compute_action_prob(self, states: np.ndarray, policy_kde: Dict) -> np.ndarray:
        """Compute P(a=1|s) using Bayes rule"""
        n = len(states)
        
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
            
            # Log probabilities
            log_p_s_a0 = policy_kde['kde_a0'].score_samples(batch_states)
            log_p_s_a1 = policy_kde['kde_a1'].score_samples(batch_states)
            
            # Priors
            log_p_a0 = np.log(1 - policy_kde['prior'] + 1e-10)
            log_p_a1 = np.log(policy_kde['prior'] + 1e-10)
            
            # Joint
            log_joint_a0 = log_p_s_a0 + log_p_a0
            log_joint_a1 = log_p_s_a1 + log_p_a1
            
            # Normalize
            log_normalizer = np.logaddexp(log_joint_a0, log_joint_a1)
            prob_a1[i:end_idx] = np.exp(log_joint_a1 - log_normalizer)
        
        return np.clip(prob_a1, 1e-6, 1-1e-6)
    
    def compute_importance_weights(self, states: np.ndarray, actions: np.ndarray,
                                  behavioral_kde: Dict, target_kde: Dict) -> np.ndarray:
        """Compute w(s,a) = p(a|s) / q(a|s)"""
        
        # Get action probabilities
        prob_behavioral = self.compute_action_prob(states, behavioral_kde)
        prob_target = self.compute_action_prob(states, target_kde)
        
        # Compute weights
        weights = np.zeros(len(states))
        
        # For a=1
        mask_a1 = (actions == 1)
        weights[mask_a1] = prob_target[mask_a1] / (prob_behavioral[mask_a1] + 1e-10)
        
        # For a=0
        mask_a0 = (actions == 0)
        weights[mask_a0] = (1 - prob_target[mask_a0]) / (1 - prob_behavioral[mask_a0] + 1e-10)
        
        return np.clip(weights, 0, 20)


# ============================================================================
# OPE Evaluator
# ============================================================================

class OPEEvaluator:
    """Off-Policy Evaluation using importance sampling"""
    
    def __init__(self, kde_bandwidth: float = 0.1):
        self.kde_sampler = KDEImportanceSampler(bandwidth=kde_bandwidth)
        
    def evaluate(self, data: DataLoader, agent: CQLAgent,
                train_mask: np.ndarray, test_mask: np.ndarray) -> Dict:
        """
        Evaluate agent using importance sampling
        
        Returns:
            Dictionary with evaluation metrics
        """
        
        # Generate learned policy actions
        print("\n  Generating learned policy actions...")
        # Process in batches for efficiency
        batch_size = 10000
        learned_actions = np.zeros(len(data.states))
        
        for i in range(0, len(data.states), batch_size):
            end_idx = min(i + batch_size, len(data.states))
            batch_states = data.states_scaled[i:end_idx]
            
            # Get actions for the batch (vectorized)
            batch_actions = agent.select_actions_batch(batch_states)
            learned_actions[i:end_idx] = batch_actions
            
            if (i // batch_size) % 2 == 0:
                print(f"    Processed {end_idx}/{len(data.states)} samples...")
            
        # Fit KDEs on training data
        print("  Fitting KDE models...")
        print(f"    Training samples: {train_mask.sum()}")
        
        # Subsample training data for faster KDE fitting
        max_kde_samples = 10000
        train_indices = np.where(train_mask)[0]
        if len(train_indices) > max_kde_samples:
            np.random.seed(42)
            train_indices = np.random.choice(train_indices, max_kde_samples, replace=False)
            print(f"    Subsampling to {max_kde_samples} samples for KDE fitting")
        
        behavioral_kde = self.kde_sampler.fit_policy(
            data.states_scaled[train_indices],
            data.actions[train_indices]
        )
        print("    Behavioral KDE fitted")
        
        target_kde = self.kde_sampler.fit_policy(
            data.states_scaled[train_indices],
            learned_actions[train_indices]
        )
        
        print(f"    Behavioral P(a=1): {behavioral_kde['prior']:.3f} (per-timestep)")
        print(f"    Learned P(a=1): {target_kde['prior']:.3f} (per-timestep)")
        
        # Also compute per-patient statistics
        if hasattr(data, 'patient_groups') and data.patient_groups:
            behavioral_patient_stats = []
            learned_patient_stats = []
            
            for patient_id, (start_idx, end_idx) in data.patient_groups.items():
                # Get actions for this patient
                patient_behavioral = data.actions[start_idx:end_idx]
                patient_learned = learned_actions[start_idx:end_idx]
                
                # Compute proportion of a=1 for this patient
                behavioral_patient_stats.append(patient_behavioral.mean())
                learned_patient_stats.append(patient_learned.mean())
            
            print(f"    Per-patient stats:")
            print(f"      Behavioral: mean={np.mean(behavioral_patient_stats):.3f}, "
                  f"std={np.std(behavioral_patient_stats):.3f}")
            print(f"      Learned: mean={np.mean(learned_patient_stats):.3f}, "
                  f"std={np.std(learned_patient_stats):.3f}")
        
        # Compute importance weights on test set
        print("  Computing importance weights...")
        weights = self.kde_sampler.compute_importance_weights(
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
# Main Evaluation Function
# ============================================================================

def evaluate_cql_model(model_path: str, model_type: str = 'binary',
                       data_path: str = 'sample_data_oviss.csv') -> Dict:
    """
    Main function to evaluate a CQL model
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('binary', 'dual_continuous', 'dual_discrete')
        data_path: Path to data file
        
    Returns:
        Dictionary with evaluation results
    """
    
    print("=" * 60)
    print("CQL OFF-POLICY EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = DataLoader(model_type)
    
    # Load model
    print("\n2. Loading CQL model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if model_type == 'binary':
        agent = BinaryCQLAgent(state_dim=data.states.shape[1])
    else:
        raise NotImplementedError(f"Model type {model_type} not yet supported")
    
    agent.load(model_path)
    
    # Split data
    print("\n3. Splitting data...")
    train_mask, test_mask = data.get_train_test_masks()
    print(f"  Train: {train_mask.sum()} samples")
    print(f"  Test: {test_mask.sum()} samples")
    
    # Evaluate
    print("\n4. Running OPE...")
    evaluator = OPEEvaluator()
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
    # Example usage
    model_path = 'experiment/binary_cql_unified_alpha01_best.pt'
    
    if os.path.exists(model_path):
        results = evaluate_cql_model(model_path, model_type='binary')
    else:
        print(f"Model not found: {model_path}")
        print("Please train the binary CQL model first")