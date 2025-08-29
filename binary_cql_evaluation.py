"""
Binary CQL Evaluation with Vasopressor Initiation Constraint
==============================================================
Once VP1 is initiated (action=1), it cannot be stopped for that patient.
This reflects clinical practice where vasopressor discontinuation is rare.

Uses the BinaryCQL class with continuous Q-network architecture from train_binary_cql.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import os
import sys

# Import BinaryCQL from train_binary_cql
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_binary_cql import BinaryCQL

from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
import data_config as config


@dataclass
class EvaluationMetrics:
    """Store evaluation metrics"""
    mortality_rate: float
    avg_vp1_usage: float
    avg_vp1_initiation_time: float
    avg_trajectory_length: float
    vp1_started_ratio: float
    never_started_ratio: float
    concordance_with_clinician: float
    avg_q_value: float
    
    def __str__(self):
        return f"""
Evaluation Metrics:
  Mortality Rate:           {self.mortality_rate*100:.1f}%
  VP1 Usage Rate:           {self.avg_vp1_usage*100:.1f}%
  VP1 Initiation Time:      {self.avg_vp1_initiation_time:.1f} hours
  VP1 Started:              {self.vp1_started_ratio*100:.1f}%
  VP1 Never Started:        {self.never_started_ratio*100:.1f}%
  Concordance w/ Clinician: {self.concordance_with_clinician*100:.1f}%
  Avg Q-value:              {self.avg_q_value:.3f}
  Avg Trajectory Length:    {self.avg_trajectory_length:.1f} timesteps
"""


class VasopressorInitiationPolicy:
    """
    Wrapper that enforces VP1 persistence constraint during evaluation.
    Once VP1 is initiated for a patient, all subsequent actions are VP1=1.
    Uses BinaryCQL with continuous Q-network architecture.
    """
    
    def __init__(self, binary_cql_agent: BinaryCQL):
        """
        Args:
            binary_cql_agent: Trained BinaryCQL agent with Q(s,a) -> R architecture
        """
        self.agent = binary_cql_agent
        self.device = self.agent.device
        self.agent.q1.eval()
        self.agent.q2.eval()
        
        # Track VP1 initiation status for each patient
        self.vp1_initiated = {}
        
    def reset(self):
        """Reset VP1 tracking for new evaluation"""
        self.vp1_initiated = {}
    
    def select_action(self, state: np.ndarray, patient_id: int, 
                     epsilon: float = 0.0) -> int:
        """
        Select action with VP1 persistence constraint
        
        Args:
            state: Current state
            patient_id: Patient identifier
            epsilon: Epsilon for epsilon-greedy (usually 0 for evaluation)
            
        Returns:
            Action (0 or 1)
        """
        # Check if VP1 already initiated for this patient
        if patient_id in self.vp1_initiated and self.vp1_initiated[patient_id]:
            return 1  # Must continue VP1
        
        # Use the agent's select_action method
        action_array = self.agent.select_action(state, epsilon=epsilon)
        action = int(action_array[0])
        
        # If VP1 is initiated, mark it for this patient
        if action == 1:
            self.vp1_initiated[patient_id] = True
        
        return action
    
    def get_q_values(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Get Q-values for both actions
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (Q(s,0), Q(s,1))
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Evaluate Q-values for both actions
            action_0 = torch.zeros(1, 1).to(self.device)
            action_1 = torch.ones(1, 1).to(self.device)
            
            # Get Q-values using min(Q1, Q2)
            q1_val_0 = self.agent.q1(state_tensor, action_0).item()
            q2_val_0 = self.agent.q2(state_tensor, action_0).item()
            q_val_0 = min(q1_val_0, q2_val_0)
            
            q1_val_1 = self.agent.q1(state_tensor, action_1).item()
            q2_val_1 = self.agent.q2(state_tensor, action_1).item()
            q_val_1 = min(q1_val_1, q2_val_1)
            
            return q_val_0, q_val_1


def evaluate_policy(
    policy: VasopressorInitiationPolicy,
    data_pipeline: IntegratedDataPipelineV2,
    split: str = 'test'
) -> EvaluationMetrics:
    """
    Evaluate policy on a dataset split
    
    Args:
        policy: Policy to evaluate
        data_pipeline: Data pipeline with prepared data
        split: Which split to evaluate on ('train', 'val', 'test')
        
    Returns:
        Evaluation metrics
    """
    print(f"\nEvaluating on {split} set...")
    
    # Get the appropriate data
    if split == 'train':
        data = data_pipeline.train_data
        patient_groups = data_pipeline.train_patient_groups
    elif split == 'val':
        data = data_pipeline.val_data
        patient_groups = data_pipeline.val_patient_groups
    else:
        data = data_pipeline.test_data
        patient_groups = data_pipeline.test_patient_groups
    
    if data is None:
        raise ValueError("Data not prepared. Call prepare_data() first.")
    
    # Reset policy tracking
    policy.reset()
    
    # Metrics to track
    total_mortality = 0
    total_vp1_usage = []
    vp1_initiation_times = []
    trajectory_lengths = []
    concordances = []
    q_values_all = []
    vp1_started_count = 0
    never_started_count = 0
    
    # Evaluate each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        traj_length = end_idx - start_idx
        trajectory_lengths.append(traj_length)
        
        # Get patient trajectory
        patient_states = data['states'][start_idx:end_idx]
        patient_clinician_actions = data['actions'][start_idx:end_idx]
        patient_rewards = data['rewards'][start_idx:end_idx]
        patient_dones = data['dones'][start_idx:end_idx]
        
        # Check mortality (negative terminal reward)
        is_died = patient_rewards[-1] < 0 and patient_dones[-1] == 1.0
        if is_died:
            total_mortality += 1
        
        # Generate policy actions for this patient
        policy_actions = []
        patient_q_values = []
        vp1_initiated_timestep = None
        
        for t in range(traj_length):
            state = patient_states[t]
            action = policy.select_action(state, patient_id, epsilon=0.0)
            policy_actions.append(action)
            
            # Track Q-values
            q_val_0, q_val_1 = policy.get_q_values(state)
            patient_q_values.append(max(q_val_0, q_val_1))
            
            # Track VP1 initiation
            if action == 1 and vp1_initiated_timestep is None:
                vp1_initiated_timestep = t
        
        policy_actions = np.array(policy_actions)
        
        # Calculate metrics for this patient
        vp1_usage = policy_actions.mean()
        total_vp1_usage.append(vp1_usage)
        
        if vp1_initiated_timestep is not None:
            vp1_initiation_times.append(vp1_initiated_timestep)
            vp1_started_count += 1
        else:
            never_started_count += 1
        
        # Concordance with clinician
        concordance = (policy_actions == patient_clinician_actions.flatten()).mean()
        concordances.append(concordance)
        
        # Store Q-values
        q_values_all.extend(patient_q_values)
    
    # Calculate aggregate metrics
    n_patients = len(patient_groups)
    
    metrics = EvaluationMetrics(
        mortality_rate=total_mortality / n_patients,
        avg_vp1_usage=np.mean(total_vp1_usage),
        avg_vp1_initiation_time=np.mean(vp1_initiation_times) if vp1_initiation_times else -1,
        avg_trajectory_length=np.mean(trajectory_lengths),
        vp1_started_ratio=vp1_started_count / n_patients,
        never_started_ratio=never_started_count / n_patients,
        concordance_with_clinician=np.mean(concordances),
        avg_q_value=np.mean(q_values_all)
    )
    
    return metrics


def load_binary_cql_model(checkpoint_path: str) -> BinaryCQL:
    """
    Load a trained Binary CQL model
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded BinaryCQL agent
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint to get state dimension
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get state dimension from checkpoint
    state_dim = checkpoint.get('state_dim', 18)  # Default to 18 for binary
    
    # Create agent
    agent = BinaryCQL(
        state_dim=state_dim,
        alpha=checkpoint.get('alpha', 1.0),
        gamma=checkpoint.get('gamma', 0.95),
        tau=checkpoint.get('tau', 0.005)
    )
    
    # Load the model
    agent.load(checkpoint_path)
    
    return agent


def demonstrate_vp1_persistence():
    """
    Demonstrate VP1 persistence with sample trajectories
    """
    print("="*70)
    print(" VP1 PERSISTENCE DEMONSTRATION")
    print("="*70)
    
    # Create a dummy agent for demonstration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_agent = BinaryCQL(state_dim=18, alpha=0.0)
    
    # Create policy wrapper
    policy = VasopressorInitiationPolicy(dummy_agent)
    
    # Simulate evaluation for 3 patients
    print("\nSimulating 3 patient trajectories:")
    print("-"*50)
    
    for patient_id in range(1, 4):
        print(f"\nPatient {patient_id}:")
        policy.reset()  # Reset for new evaluation batch
        
        actions = []
        q_values = []
        for timestep in range(10):
            # Create random state
            state = np.random.randn(18)
            
            # Get action with persistence
            action = policy.select_action(state, patient_id, epsilon=0.1)
            actions.append(action)
            
            # Get Q-values
            q_val_0, q_val_1 = policy.get_q_values(state)
            q_values.append((q_val_0, q_val_1))
            
            print(f"  t={timestep}: action={action}, Q(s,0)={q_val_0:.3f}, Q(s,1)={q_val_1:.3f}", end="")
            if patient_id in policy.vp1_initiated and policy.vp1_initiated[patient_id]:
                print(" (VP1 locked ON)")
            else:
                print(" (VP1 can change)")
        
        print(f"  Actions: {actions}")
        print(f"  VP1 usage: {np.mean(actions)*100:.1f}%")
        
        # Check persistence property
        if 1 in actions:
            first_vp1_idx = actions.index(1)
            remaining_actions = actions[first_vp1_idx:]
            if all(a == 1 for a in remaining_actions):
                print("  ✅ VP1 persistence maintained")
            else:
                print("  ❌ VP1 persistence violated!")


def main():
    """
    Main evaluation function
    """
    print("="*70)
    print(" BINARY CQL EVALUATION WITH VP1 PERSISTENCE")
    print("="*70)
    
    # Initialize data pipeline
    print("\nPreparing data pipeline...")
    pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Try to load a trained model
    model_paths = [
        'experiment/binary_cql_continuous_alpha1.0_best.pt',
        'experiment/binary_cql_continuous_alpha1.0_final.pt',
        'experiment/binary_cql_alpha01_best.pt',
        'experiment/binary_cql_alpha01_final.pt',
        'experiment/binary_cql_alpha00_best.pt',
    ]
    
    agent = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"\nLoading model from: {path}")
            try:
                agent = load_binary_cql_model(path)
                print("✅ Model loaded successfully")
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    if agent is None:
        print("\n⚠️ No trained model found, creating random agent for demonstration")
        agent = BinaryCQL(state_dim=18, alpha=1.0)
    
    # Create policy with VP1 persistence
    policy = VasopressorInitiationPolicy(agent)
    
    # Evaluate on each split
    for split in ['train', 'val', 'test']:
        metrics = evaluate_policy(policy, pipeline, split)
        print(f"\n{'='*50}")
        print(f" {split.upper()} SET RESULTS")
        print(f"{'='*50}")
        print(metrics)
    
    # Demonstrate VP1 persistence
    print("\n" + "="*70)
    demonstrate_vp1_persistence()
    
    return policy, pipeline


if __name__ == "__main__":
    policy, pipeline = main()