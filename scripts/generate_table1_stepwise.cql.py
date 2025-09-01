#!/usr/bin/env python3
"""
Generate Table 1: Comprehensive comparison of CQL models
Outputs LaTeX table directly
"""

import numpy as np
import torch
import os
import sys

# Add parent directory to path to import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from train_binary_cql import BinaryCQL
from run_unified_cql_allalphas import DualMixedCQL  # Changed from StableCQL
from run_block_discrete_cql_allalphas import DualBlockDiscreteCQL
from run_unified_stepwise_cql_allalphas import UnifiedStepwiseCQL, StepwiseActionSpace
from vaso_init_policy import VasopressorInitiationPolicy, DualVasopressorInitiationPolicy


class StepwiseVasopressorPolicy:
    """
    Policy wrapper for stepwise CQL that:
    1. Maintains VP1 persistence (once started, cannot stop)
    2. Tracks VP2 trajectory across timesteps
    3. Properly handles state augmentation with VP2 one-hot encoding
    """
    
    def __init__(self, agent, action_space):
        self.agent = agent
        self.action_space = action_space
        self.vp1_active = {}  # Track VP1 status per patient
        self.current_vp2 = {}  # Track current VP2 dose per patient
        
    def reset(self):
        """Reset for new patient"""
        # Don't clear dictionaries, just let new patients be added
        pass
    
    def select_action(self, state, patient_id):
        """
        Select action with VP1 persistence and VP2 tracking
        
        Args:
            state: Base state from pipeline (17 dims)
            patient_id: Patient identifier
            
        Returns:
            [vp1, vp2] continuous action values
        """
        # Initialize patient if first time
        if patient_id not in self.vp1_active:
            self.vp1_active[patient_id] = False
            self.current_vp2[patient_id] = 0.0
        
        # Get current VP2 dose for this patient
        current_vp2 = self.current_vp2[patient_id]
        
        # Augment state with VP2 one-hot encoding
        vp2_one_hot = self.action_space.vp2_to_one_hot(current_vp2)
        augmented_state = np.concatenate([state, vp2_one_hot])
        state_tensor = torch.FloatTensor(augmented_state).to(self.agent.device)
        
        # Get best discrete action from agent
        discrete_action = self.agent.select_action(state_tensor, current_vp2)
        
        # Apply action to get new VP1 and VP2 values
        vp1_value, new_vp2 = self.action_space.apply_action(discrete_action, current_vp2)
        
        # Apply VP1 persistence
        if self.vp1_active[patient_id]:
            vp1_value = 1.0
        elif vp1_value > 0:
            self.vp1_active[patient_id] = True
        
        # Update tracked VP2 dose
        self.current_vp2[patient_id] = new_vp2
        
        return np.array([vp1_value, new_vp2])
    
    def get_q_value(self, state, action):
        """
        Get Q-value for state-action pair
        
        Args:
            state: Base state (17 dims)
            action: [vp1, vp2] continuous values
            
        Returns:
            Q-value
        """
        # Get VP2 from action
        current_vp2 = action[1]
        
        # Augment state with VP2 one-hot
        vp2_one_hot = self.action_space.vp2_to_one_hot(current_vp2)
        augmented_state = np.concatenate([state, vp2_one_hot])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(augmented_state).unsqueeze(0).to(self.agent.device)
            
            # Need to find appropriate discrete action
            # For simplicity, use no-change action for VP2
            vp1_binary = int(action[0] > 0.5)
            vp2_change_idx = len(self.action_space.VP2_CHANGES) // 2  # Middle index = no change
            discrete_action = vp1_binary * self.action_space.n_vp2_actions + vp2_change_idx  # Shape: (batch_size,)
            #discrete_action = self.action_space.get_discrete_action(vp1_binary, vp2_change_idx)
            action_tensor = torch.LongTensor([discrete_action]).to(self.agent.device)
            
            q1 = self.agent.q1(state_tensor, action_tensor).item()
            q2 = self.agent.q2(state_tensor, action_tensor).item()
            
        return min(q1, q2)


def evaluate_model(model_type='binary', alpha=0.001, apply_persistence=False, vp2_bins=None, max_step=None):
    """Evaluate a model on test set"""
    
    # Map alpha to file suffix
    if alpha == 0.001:
        suffix = "alpha001"
    elif alpha == 0.01:
        suffix = "alpha01"
    else:
        suffix = f"alpha{str(alpha).replace('.', '')}"
    
    # Initialize pipeline
    # Block discrete and stepwise use dual data type
    data_type = 'dual' if model_type in ['dual', 'block_discrete', 'stepwise'] else model_type
    pipeline = IntegratedDataPipelineV2(model_type=data_type, random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Initialize action_space for stepwise models (will be None for others)
    action_space = None
    
    # Load model
    if model_type == 'binary':
        # Use new naming convention for binary models
        if alpha == 0.0: 
            astr = '00'
        else: 
            astr = str(alpha).split('.')[1]
        model_path = f'experiment/binary_cql_unified_alpha{astr}_best.pt'
        
        agent = BinaryCQL(state_dim=18, alpha=alpha, tau=0.8, lr=1e-3)
        agent.load(model_path)
        
        if apply_persistence:
            policy = VasopressorInitiationPolicy(agent)
        else:
            policy = None
    elif model_type == 'block_discrete':
        # Block discrete model
        model_path = f'experiment/block_discrete_cql_alpha{alpha:.4f}_bins{vp2_bins}_best.pt'
        
        checkpoint = torch.load(model_path, map_location='cuda')
        agent = DualBlockDiscreteCQL(
            state_dim=17, 
            vp2_bins=vp2_bins,
            alpha=alpha, 
            tau=0.8, 
            lr=1e-3
        )
        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        if apply_persistence:
            policy = DualVasopressorInitiationPolicy(agent)
        else:
            policy = None
    elif model_type == 'stepwise':
        # Stepwise CQL model
        model_path = f'experiment/stepwise_cql_alpha{alpha:.6f}_maxstep{max_step:.1f}_best.pt'
        
        checkpoint = torch.load(model_path, map_location='cuda')
        
        # Initialize action space (needed for state dimension)
        action_space = StepwiseActionSpace(max_step=max_step)
        base_state_dim = 17  # From IntegratedDataPipelineV2 dual mode
        total_state_dim = base_state_dim + action_space.n_vp2_bins  # 17 + 10 = 27
        
        # Initialize agent
        agent = UnifiedStepwiseCQL(
            state_dim=total_state_dim,
            max_step=max_step,
            alpha=alpha,
            tau=0.8,
            lr=1e-3
        )
        
        # Load model weights
        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        # Stepwise requires special policy wrapper
        if apply_persistence:
            # Will need to create StepwiseVasopressorPolicy
            policy = StepwiseVasopressorPolicy(agent, action_space)
        else:
            policy = None
            
    else:
        # Use new naming convention for dual mixed models
        model_path = f'experiment/dual_rev_cql_alpha{alpha:.4f}_best.pt'
        
        checkpoint = torch.load(model_path, map_location='cuda')
        # DualMixedCQL takes state_dim as parameter (should match training data)
        agent = DualMixedCQL(state_dim=17, alpha=alpha, tau=0.8, lr=1e-3)
        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        if apply_persistence:
            policy = DualVasopressorInitiationPolicy(agent)
        else:
            policy = None
    
    # Evaluate on test data
    patient_groups = test_data['patient_groups']
    
    vp1_usage = []
    vp2_usage = []
    q_values_timestep = []
    q_values_patient = []
    vp1_concordances = []
    vp2_concordances = []  # For block discrete models only
    delta_q_timestep = []  # Delta Q per timestep
    delta_q_patient = []  # Delta Q per patient
    
    # Process each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        patient_states = test_data['states'][start_idx:end_idx]
        patient_actions = test_data['actions'][start_idx:end_idx]
        
        patient_q_values = []
        patient_delta_q = []  # Track delta Q for this patient
        patient_vp1_concordance = []
        patient_vp2_concordance = []  # For block discrete only
        vp1_used = False
        vp2_used = False
        
        if policy:
            policy.reset()  # Reset for each patient
        
        for t in range(len(patient_states)):
            state = patient_states[t]
            clinician_action = patient_actions[t]
            
            if model_type == 'binary':
                if policy:
                    model_action = policy.select_action(state, patient_id, epsilon=0.0)
                    q0, q1 = policy.get_q_values(state)
                    q_val = max(q0, q1)
                    # Get Q-value for clinician action
                    q_clinician = q0 if clinician_action == 0 else q1
                else:
                    model_action_arr = agent.select_action(state, epsilon=0.0)
                    model_action = int(model_action_arr[0])
                    
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        # Q-value for model's optimal action
                        action_t = torch.FloatTensor([model_action]).unsqueeze(0).to(agent.device)
                        q1 = agent.q1(state_t, action_t).item()
                        q2 = agent.q2(state_t, action_t).item()
                        q_val = min(q1, q2)
                        
                        # Q-value for clinician's action
                        clinician_action_t = torch.FloatTensor([clinician_action]).unsqueeze(0).to(agent.device)
                        q1_clin = agent.q1(state_t, clinician_action_t).item()
                        q2_clin = agent.q2(state_t, clinician_action_t).item()
                        q_clinician = min(q1_clin, q2_clin)
                
                # Calculate delta Q (model optimal - clinician)
                delta_q = q_val - q_clinician
                patient_delta_q.append(delta_q)
                delta_q_timestep.append(delta_q)
                
                if model_action > 0:
                    vp1_used = True
                patient_vp1_concordance.append(model_action == clinician_action)
                
            else:  # Dual, Block Discrete, or Stepwise
                if policy:
                    # Use policy wrapper with persistence
                    model_action = policy.select_action(state, patient_id)
                    q_val = policy.get_q_value(state, model_action)
                    # Get Q-value for clinician action
                    q_clinician = policy.get_q_value(state, clinician_action)
                else:
                    # Use agent directly (only for non-stepwise)
                    if model_type == 'stepwise':
                        # Stepwise requires tracking VP2 trajectory, must use policy
                        raise ValueError("Stepwise model requires policy wrapper for evaluation")
                    
                    model_action = agent.select_action(state)
                    
                    if model_type == 'block_discrete':
                        # For block discrete, need to get Q-value differently
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                            # Q-value for model's optimal action
                            action_idx = agent.continuous_to_discrete_action(model_action)
                            action_idx_t = torch.LongTensor([action_idx]).to(agent.device)
                            q1 = agent.q1(state_t, action_idx_t).item()
                            q2 = agent.q2(state_t, action_idx_t).item()
                            q_val = min(q1, q2)
                            
                            # Q-value for clinician's action
                            clinician_idx = agent.continuous_to_discrete_action(clinician_action)
                            clinician_idx_t = torch.LongTensor([clinician_idx]).to(agent.device)
                            q1_clin = agent.q1(state_t, clinician_idx_t).item()
                            q2_clin = agent.q2(state_t, clinician_idx_t).item()
                            q_clinician = min(q1_clin, q2_clin)
                    else:
                        # Dual mixed model
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                            # Q-value for model's optimal action
                            action_t = torch.FloatTensor(model_action).unsqueeze(0).to(agent.device)
                            q1 = agent.q1(state_t, action_t).item()
                            q2 = agent.q2(state_t, action_t).item()
                            q_val = min(q1, q2)
                            
                            # Q-value for clinician's action
                            clinician_action_t = torch.FloatTensor(clinician_action).unsqueeze(0).to(agent.device)
                            q1_clin = agent.q1(state_t, clinician_action_t).item()
                            q2_clin = agent.q2(state_t, clinician_action_t).item()
                            q_clinician = min(q1_clin, q2_clin)
                
                # Calculate delta Q
                delta_q = q_val - q_clinician
                patient_delta_q.append(delta_q)
                delta_q_timestep.append(delta_q)
                
                # VP1 is binary (0 or 1), VP2 is continuous [0, 0.5]
                if model_action[0] > 0:  # VP1 is binary
                    vp1_used = True
                if model_action[1] > 0.01:  # VP2 threshold
                    vp2_used = True
                    
                # VP1 Concordance (binary comparison)
                clinician_vp1 = clinician_action[0] if clinician_action.ndim > 0 else clinician_action
                patient_vp1_concordance.append((model_action[0] > 0) == (clinician_vp1 > 0))
                
                # VP2 Concordance for Block Discrete and Stepwise models
                if model_type == 'block_discrete':
                    clinician_vp2 = clinician_action[1] if len(clinician_action) > 1 else 0.0
                    
                    # Find which bin the clinician's VP2 falls into
                    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
                    clinician_vp2_bin = np.digitize(clinician_vp2, vp2_bin_edges) - 1
                    clinician_vp2_bin = np.clip(clinician_vp2_bin, 0, vp2_bins - 1)
                    
                    # Find which bin the model's VP2 falls into
                    model_vp2_bin = np.digitize(model_action[1], vp2_bin_edges) - 1
                    model_vp2_bin = np.clip(model_vp2_bin, 0, vp2_bins - 1)
                    
                    # Check concordance
                    patient_vp2_concordance.append(model_vp2_bin == clinician_vp2_bin)
                    
                elif model_type == 'stepwise':
                    clinician_vp2 = clinician_action[1] if len(clinician_action) > 1 else 0.0
                    
                    # For stepwise, use the action space's VP2 bins
                    # The stepwise model discretizes VP2 into bins [0.05, 0.10, ..., 0.50]
                    clinician_vp2_bin = action_space.vp2_to_bin(clinician_vp2)
                    model_vp2_bin = action_space.vp2_to_bin(model_action[1])
                    
                    # Check concordance
                    patient_vp2_concordance.append(model_vp2_bin == clinician_vp2_bin)
            
            patient_q_values.append(q_val)
            q_values_timestep.append(q_val)
        
        vp1_usage.append(vp1_used)
        if model_type in ['dual', 'block_discrete', 'stepwise']:
            vp2_usage.append(vp2_used)
        q_values_patient.append(np.mean(patient_q_values))
        delta_q_patient.append(np.mean(patient_delta_q))  # Add patient-level delta Q
        vp1_concordances.append(np.mean(patient_vp1_concordance))
        if model_type in ['block_discrete', 'stepwise'] and patient_vp2_concordance:
            vp2_concordances.append(np.mean(patient_vp2_concordance))
    
    # Calculate metrics
    results = {
        'vp1_usage': np.mean(vp1_usage) * 100,
        'vp2_usage': np.mean(vp2_usage) * 100 if vp2_usage else None,
        'q_per_timestep': np.mean(q_values_timestep),
        'q_per_patient': np.mean(q_values_patient),
        'delta_q_per_timestep': np.mean(delta_q_timestep),  # Average delta Q per timestep
        'delta_q_per_patient': np.mean(delta_q_patient),  # Average delta Q per patient
        'vp1_concordance': np.mean(vp1_concordances) * 100,
        'vp2_concordance': np.mean(vp2_concordances) * 100 if vp2_concordances else None
    }
    
    return results


def generate_latex_table(max_step):
    """Generate LaTeX table comparing Clinician, Binary CQL, and Stepwise CQL models"""
    
    print("="*70)
    print(" STEPWISE CQL MODEL COMPARISON")
    print("="*70)
    
    # Evaluate models
    results = {}
    
    # Binary CQL with persistence (alpha=0.001 as baseline)
    print(f"\nEvaluating Binary CQL (alpha=0.001)...")
    results['binary'] = evaluate_model('binary', alpha=0.001, apply_persistence=True)
    
    # Stepwise CQL models - including all requested alphas
    print(f"\nEvaluating Stepwise CQL (alpha=0.000000)...")
    results['stepwise_0.000000'] = evaluate_model('stepwise', alpha=0.000000, apply_persistence=True, max_step=max_step)
    
    print(f"\nEvaluating Stepwise CQL (alpha=0.000100)...")
    results['stepwise_0.000100'] = evaluate_model('stepwise', alpha=0.000100, apply_persistence=True, max_step=max_step)
    
    print(f"\nEvaluating Stepwise CQL (alpha=0.001000)...")
    results['stepwise_0.001000'] = evaluate_model('stepwise', alpha=0.001000, apply_persistence=True, max_step=max_step)
    
    print(f"\nEvaluating Stepwise CQL (alpha=0.010000)...")
    results['stepwise_0.010000'] = evaluate_model('stepwise', alpha=0.010000, apply_persistence=True, max_step=max_step)
    
    # Helper function to format values
    def fmt(val, fmt_str=".1f"):
        if val is None:
            return "N/A"
        try:
            return f"{val:{fmt_str}}"
        except:
            return "N/A"
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{Comparison of Binary CQL and Stepwise CQL models with vasopressor persistence policy}
\label{tab:stepwise_comparison}
\begin{tabular}{lccccccc}
\toprule
Model & $\alpha$ & VP1 (\%) & VP2 (\%) & Q/time & $\Delta$Q & VP1 C. (\%) & VP2 C. (\%) \\
\midrule
Clinician & -- & 38.8 & 99.8 & 0.000 & 0.000 & 100.0 & -- \\
\midrule"""
    
    # Add model results
    b_res = results['binary']
    s0_res = results['stepwise_0.000000']
    s1_res = results['stepwise_0.000100']
    s2_res = results['stepwise_0.001000']
    s3_res = results['stepwise_0.010000']
    
    # Binary CQL
    latex += f"""
Binary CQL & 0.001 & {fmt(b_res.get('vp1_usage'))} & -- & {fmt(b_res.get('q_per_timestep'), '.3f')} & {fmt(b_res.get('delta_q_per_timestep'), '.3f')} & {fmt(b_res.get('vp1_concordance'))} & -- \\\\"""
    
    # Stepwise CQL models
    latex += f"""
Stepwise CQL & 0.000000 & {fmt(s0_res.get('vp1_usage'))} & {fmt(s0_res.get('vp2_usage'))} & {fmt(s0_res.get('q_per_timestep'), '.3f')} & {fmt(s0_res.get('delta_q_per_timestep'), '.3f')} & {fmt(s0_res.get('vp1_concordance'))} & {fmt(s0_res.get('vp2_concordance'))} \\\\
Stepwise CQL & 0.000100 & {fmt(s1_res.get('vp1_usage'))} & {fmt(s1_res.get('vp2_usage'))} & {fmt(s1_res.get('q_per_timestep'), '.3f')} & {fmt(s1_res.get('delta_q_per_timestep'), '.3f')} & {fmt(s1_res.get('vp1_concordance'))} & {fmt(s1_res.get('vp2_concordance'))} \\\\
Stepwise CQL & 0.001000 & {fmt(s2_res.get('vp1_usage'))} & {fmt(s2_res.get('vp2_usage'))} & {fmt(s2_res.get('q_per_timestep'), '.3f')} & {fmt(s2_res.get('delta_q_per_timestep'), '.3f')} & {fmt(s2_res.get('vp1_concordance'))} & {fmt(s2_res.get('vp2_concordance'))} \\\\
Stepwise CQL & 0.010000 & {fmt(s3_res.get('vp1_usage'))} & {fmt(s3_res.get('vp2_usage'))} & {fmt(s3_res.get('q_per_timestep'), '.3f')} & {fmt(s3_res.get('delta_q_per_timestep'), '.3f')} & {fmt(s3_res.get('vp1_concordance'))} & {fmt(s3_res.get('vp2_concordance'))} \\\\"""
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open(f'stepwise_v2_deltaq_max_step{max_step:.1f}_comparison_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\n✓ LaTeX table saved to: stepwise_v2_deltaq_max_step{max_step:.1f}_comparison_table.tex")
    
    # Also print results
    print("\nNumerical Results:")
    print("-"*80)
    print(f"{'Model':<20} {'Alpha':<8} {'VP1 (%)':<10} {'VP2 (%)':<10} {'Q/time':<10} {'ΔQ':<10} {'VP1-C (%)':<10} {'VP2-C (%)':<10}")
    print("-"*90)
    print(f"{'Clinician':<20} {'--':<8} {'38.8':<10} {'99.8':<10} {'0.000':<10} {'0.000':<10} {'100.0':<10} {'--':<10}")
    print(f"{'Binary CQL':<20} {'0.001':<8} {fmt(b_res.get('vp1_usage')):<10} {'--':<10} {fmt(b_res.get('q_per_timestep'), '.3f'):<10} {fmt(b_res.get('delta_q_per_timestep'), '.3f'):<10} {fmt(b_res.get('vp1_concordance')):<10} {'--':<10}")
    print(f"{'Stepwise CQL':<20} {'0.000000':<8} {fmt(s0_res.get('vp1_usage')):<10} {fmt(s0_res.get('vp2_usage')):<10} {fmt(s0_res.get('q_per_timestep'), '.3f'):<10} {fmt(s0_res.get('delta_q_per_timestep'), '.3f'):<10} {fmt(s0_res.get('vp1_concordance')):<10} {fmt(s0_res.get('vp2_concordance')):<10}")
    print(f"{'Stepwise CQL':<20} {'0.000100':<8} {fmt(s1_res.get('vp1_usage')):<10} {fmt(s1_res.get('vp2_usage')):<10} {fmt(s1_res.get('q_per_timestep'), '.3f'):<10} {fmt(s1_res.get('delta_q_per_timestep'), '.3f'):<10} {fmt(s1_res.get('vp1_concordance')):<10} {fmt(s1_res.get('vp2_concordance')):<10}")
    print(f"{'Stepwise CQL':<20} {'0.001000':<8} {fmt(s2_res.get('vp1_usage')):<10} {fmt(s2_res.get('vp2_usage')):<10} {fmt(s2_res.get('q_per_timestep'), '.3f'):<10} {fmt(s2_res.get('delta_q_per_timestep'), '.3f'):<10} {fmt(s2_res.get('vp1_concordance')):<10} {fmt(s2_res.get('vp2_concordance')):<10}")
    print(f"{'Stepwise CQL':<20} {'0.010000':<8} {fmt(s3_res.get('vp1_usage')):<10} {fmt(s3_res.get('vp2_usage')):<10} {fmt(s3_res.get('q_per_timestep'), '.3f'):<10} {fmt(s3_res.get('delta_q_per_timestep'), '.3f'):<10} {fmt(s3_res.get('vp1_concordance')):<10} {fmt(s3_res.get('vp2_concordance')):<10}")
    
    return results


if __name__ == "__main__":
    max_step = 0.2
    results = generate_latex_table(max_step)
    print("\n✅ Stepwise CQL comparison complete!")