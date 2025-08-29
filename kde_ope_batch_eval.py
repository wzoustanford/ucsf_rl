#!/usr/bin/env python3
"""
Batch KDE Off-Policy Evaluation for all CQL models
===================================================

Evaluates all models listed in experiment/kde_ope_eval_list.txt using
KDE-based importance sampling for off-policy evaluation.

Supports:
- Binary CQL models
- Dual (mixed) continuous CQL models  
- Block discrete CQL models with LSTM
- LSTM Binary CQL models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import data pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

# Import model architectures
from train_binary_cql import BinaryCQL
from train_cql_stable import StableCQL
from run_lstm_block_discrete_cql_with_logging import LSTMBlockDiscreteCQL, LSTMBlockDiscreteQNetwork
from lstm_cql_network import LSTMBinaryCQL
from medical_sequence_buffer import MedicalSequenceBuffer, SequenceDataLoader


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
            
        # Process in batches for efficiency
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
        
        # Clip for stability
        return np.clip(weights, 0, 20)


# ============================================================================
# Model Loaders
# ============================================================================

def load_binary_cql(model_path: str, state_dim: int = 18):
    """Load Binary CQL model"""
    agent = BinaryCQL(state_dim=state_dim)
    agent.load(model_path)
    return agent


def load_dual_cql(model_path: str, state_dim: int = 18):
    """Load Dual Continuous CQL model"""
    agent = StableCQL(state_dim=state_dim, action_dim=2)
    
    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
    agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    
    return agent


def load_block_discrete_cql(model_path: str, state_dim: int = 18, vp2_bins: int = None):
    """Load Block Discrete CQL model with LSTM"""
    # Extract bins from filename if not provided
    if vp2_bins is None:
        import re
        match = re.search(r'bins(\d+)', model_path)
        if match:
            vp2_bins = int(match.group(1))
        else:
            vp2_bins = 5  # default
    
    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize agent with saved hyperparameters
    agent = LSTMBlockDiscreteCQL(
        state_dim=state_dim,
        vp2_bins=vp2_bins,
        hidden_dim=checkpoint.get('hidden_dim', 64),
        lstm_hidden=checkpoint.get('lstm_hidden', 64),
        num_lstm_layers=checkpoint.get('num_lstm_layers', 2),
        alpha=checkpoint.get('alpha', 0.01),
        gamma=checkpoint.get('gamma', 0.95),
        tau=checkpoint.get('tau', 0.8),
        lr=checkpoint.get('lr', 1e-3)
    )
    
    # Load model weights
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
    agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    
    return agent


def load_lstm_binary_cql(model_path: str, state_dim: int = 18):
    """Load LSTM Binary CQL model"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize agent
    agent = LSTMBinaryCQL(
        state_dim=state_dim,
        hidden_dim=checkpoint.get('hidden_dim', 128),
        lstm_hidden=checkpoint.get('lstm_hidden', 64),
        num_lstm_layers=checkpoint.get('num_lstm_layers', 2),
        alpha=checkpoint.get('alpha', 0.0),
        gamma=checkpoint.get('gamma', 0.95),
        tau=checkpoint.get('tau', 0.005),
        lr=checkpoint.get('lr', 1e-4)
    )
    
    # Load model weights
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
    agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    
    return agent


def get_model_type(model_name: str) -> str:
    """Determine model type from filename"""
    if 'block_discrete' in model_name:
        return 'block_discrete'
    elif 'lstm_cql' in model_name:
        return 'lstm_binary'
    elif 'binary_cql' in model_name:
        return 'binary'
    elif 'dual_cql' in model_name:
        return 'dual'
    else:
        raise ValueError(f"Unknown model type for {model_name}")


def get_learned_actions(agent, states: np.ndarray, model_type: str, 
                        use_initiation_policy: bool = True) -> np.ndarray:
    """Get learned policy actions for different model types"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type in ['binary', 'lstm_binary']:
        # Binary action selection
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(device)
            
            if hasattr(agent, 'select_actions_batch'):
                actions = agent.select_actions_batch(states_tensor).cpu().numpy().squeeze()
            else:
                # Manual batch selection
                batch_size = len(states)
                actions_0 = torch.zeros(batch_size, 1).to(device)
                actions_1 = torch.ones(batch_size, 1).to(device)
                
                q1_a0 = agent.q1(states_tensor, actions_0).squeeze()
                q2_a0 = agent.q2(states_tensor, actions_0).squeeze()
                q_a0 = torch.min(q1_a0, q2_a0)
                
                q1_a1 = agent.q1(states_tensor, actions_1).squeeze()
                q2_a1 = agent.q2(states_tensor, actions_1).squeeze()
                q_a1 = torch.min(q1_a1, q2_a1)
                
                actions = (q_a1 > q_a0).float().cpu().numpy()
        
        # Apply initiation policy if requested
        if use_initiation_policy:
            # This should be done per-patient trajectory
            # For now, simplified version
            initiated = False
            for i in range(len(actions)):
                if initiated:
                    actions[i] = 1.0
                elif actions[i] == 1.0:
                    initiated = True
                    
    elif model_type == 'dual':
        # For dual model, we need to discretize to binary for KDE
        # Use VP1 component (action[0]) and threshold at 0.5
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(device)
            
            if hasattr(agent, 'select_actions'):
                continuous_actions = []
                batch_size = 1000
                for i in range(0, len(states), batch_size):
                    batch_states = states_tensor[i:i+batch_size]
                    batch_actions = agent.actor(batch_states).cpu().numpy()
                    continuous_actions.append(batch_actions)
                continuous_actions = np.vstack(continuous_actions)
            else:
                # Fallback: sample actions
                continuous_actions = np.random.rand(len(states), 2)
                continuous_actions[:, 1] *= 0.5  # VP2 is 0-0.5
        
        # Convert to binary based on VP1
        actions = (continuous_actions[:, 0] > 0.5).astype(float)
        
    elif model_type == 'block_discrete':
        # For block discrete, convert to binary based on VP1
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(device)
            
            if hasattr(agent, 'select_actions_batch'):
                discrete_actions = agent.select_actions_batch(states_tensor)
            else:
                # Get Q-values for all actions
                batch_size = len(states)
                q_values = []
                
                for action_idx in range(agent.total_actions):
                    q1_vals = agent.q1(states_tensor).squeeze()[:, action_idx]
                    q2_vals = agent.q2(states_tensor).squeeze()[:, action_idx]
                    q_vals = torch.min(q1_vals, q2_vals)
                    q_values.append(q_vals)
                
                q_values = torch.stack(q_values, dim=1)
                best_actions = q_values.argmax(dim=1).cpu().numpy()
                
                # Convert to VP1 binary (action < vp2_bins means VP1=0)
                discrete_actions = (best_actions >= agent.vp2_bins).astype(float)
        
        actions = discrete_actions
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return actions


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_model(model_path: str, pipeline, kde_sampler, results_dict: Dict):
    """Evaluate a single model using KDE OPE"""
    
    model_name = os.path.basename(model_path)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Check if model exists
    full_path = os.path.join('experiment', model_name)
    if not os.path.exists(full_path):
        print(f"  ERROR: Model not found at {full_path}")
        results_dict[model_name] = {'error': 'Model file not found'}
        return
    
    try:
        # Determine model type
        model_type = get_model_type(model_name)
        print(f"  Model type: {model_type}")
        
        # Load model
        if model_type == 'binary':
            agent = load_binary_cql(full_path)
        elif model_type == 'dual':
            agent = load_dual_cql(full_path)
        elif model_type == 'block_discrete':
            agent = load_block_discrete_cql(full_path)
        elif model_type == 'lstm_binary':
            agent = load_lstm_binary_cql(full_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"  Model loaded successfully")
        
        # Get data
        train_data, val_data, test_data = pipeline.prepare_data()
        
        # Combine train+val for KDE fitting, test for evaluation
        train_states = np.vstack([train_data['states'], val_data['states']])
        train_actions = np.concatenate([train_data['actions'], val_data['actions']])
        
        test_states = test_data['states']
        test_actions = test_data['actions']
        test_rewards = test_data['rewards']
        
        # For binary models, ensure actions are binary
        if model_type in ['binary', 'lstm_binary', 'block_discrete']:
            train_actions = (train_actions > 0.5).astype(float) if train_actions.ndim == 1 else train_actions[:, 0]
            test_actions = (test_actions > 0.5).astype(float) if test_actions.ndim == 1 else test_actions[:, 0]
        elif model_type == 'dual':
            # Use VP1 component for binary classification
            train_actions = train_actions[:, 0] if train_actions.ndim > 1 else train_actions
            test_actions = test_actions[:, 0] if test_actions.ndim > 1 else test_actions
            train_actions = (train_actions > 0.5).astype(float)
            test_actions = (test_actions > 0.5).astype(float)
        
        # Generate learned policy actions
        print(f"  Generating learned policy actions...")
        use_initiation = model_type in ['binary', 'lstm_binary']
        learned_train_actions = get_learned_actions(agent, train_states, model_type, use_initiation)
        learned_test_actions = get_learned_actions(agent, test_states, model_type, use_initiation)
        
        # Subsample for KDE fitting
        max_kde_samples = 10000
        if len(train_states) > max_kde_samples:
            np.random.seed(42)
            idx = np.random.choice(len(train_states), max_kde_samples, replace=False)
            kde_train_states = train_states[idx]
            kde_train_actions = train_actions[idx]
            kde_learned_actions = learned_train_actions[idx]
        else:
            kde_train_states = train_states
            kde_train_actions = train_actions
            kde_learned_actions = learned_train_actions
        
        # Fit KDEs
        print(f"  Fitting KDE models...")
        behavioral_kde = kde_sampler.fit_policy(kde_train_states, kde_train_actions)
        learned_kde = kde_sampler.fit_policy(kde_train_states, kde_learned_actions)
        
        print(f"    Behavioral P(a=1): {behavioral_kde['prior']:.3f}")
        print(f"    Learned P(a=1): {learned_kde['prior']:.3f}")
        
        # Compute importance weights
        print(f"  Computing importance weights...")
        weights = kde_sampler.compute_importance_weights(
            test_states, test_actions, behavioral_kde, learned_kde
        )
        
        # Compute OPE metrics
        behavioral_reward = test_rewards.mean()
        is_estimate = np.mean(weights * test_rewards)
        wis_estimate = np.sum(weights * test_rewards) / np.sum(weights)
        ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
        ess_ratio = ess / len(test_rewards)
        
        # Store results
        results = {
            'model_type': model_type,
            'behavioral_reward': float(behavioral_reward),
            'is_estimate': float(is_estimate),
            'wis_estimate': float(wis_estimate),
            'ess': float(ess),
            'ess_ratio': float(ess_ratio),
            'weight_mean': float(weights.mean()),
            'weight_std': float(weights.std()),
            'weight_max': float(weights.max()),
            'behavioral_p_a1': float(behavioral_kde['prior']),
            'learned_p_a1': float(learned_kde['prior']),
            'learned_p_a1_test': float(learned_test_actions.mean())
        }
        
        results_dict[model_name] = results
        
        print(f"\n  Results:")
        print(f"    Behavioral reward: {behavioral_reward:.4f}")
        print(f"    IS estimate: {is_estimate:.4f}")
        print(f"    WIS estimate: {wis_estimate:.4f}")
        print(f"    ESS: {ess:.1f} ({ess_ratio*100:.1f}%)")
        
    except Exception as e:
        print(f"  ERROR evaluating {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        results_dict[model_name] = {'error': str(e)}


def main():
    """Main evaluation loop"""
    
    print("="*70)
    print("KDE OFF-POLICY EVALUATION - BATCH MODE")
    print("="*70)
    
    # Read model list
    model_list_file = 'experiment/kde_ope_eval_list.txt'
    with open(model_list_file, 'r') as f:
        model_names = [line.strip() for line in f if line.strip()]
    
    print(f"\nFound {len(model_names)} models to evaluate")
    
    # Initialize data pipeline (using binary for consistency)
    print("\nInitializing data pipeline...")
    pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    
    # Initialize KDE sampler
    kde_sampler = KDEImportanceSampler(bandwidth=0.1)
    
    # Results storage
    all_results = {}
    
    # Evaluate each model
    for model_name in model_names:
        evaluate_model(model_name, pipeline, kde_sampler, all_results)
    
    # Save results
    output_file = 'experiment/kde_ope_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")
    
    # Print summary table
    print("\nSUMMARY TABLE:")
    print(f"{'Model':<50} {'Type':<15} {'WIS':<8} {'ESS%':<8}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            print(f"{model_name:<50} {results['model_type']:<15} "
                  f"{results['wis_estimate']:<8.4f} {results['ess_ratio']*100:<8.1f}")
        else:
            print(f"{model_name:<50} ERROR: {results['error'][:30]}")


if __name__ == "__main__":
    main()