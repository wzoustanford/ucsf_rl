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
from vaso_init_policy import VasopressorInitiationPolicy, DualVasopressorInitiationPolicy


def evaluate_model(model_type='binary', alpha=0.001, apply_persistence=False, vp2_bins=None):
    """Evaluate a model on test set"""
    
    # Map alpha to file suffix
    if alpha == 0.001:
        suffix = "alpha001"
    elif alpha == 0.01:
        suffix = "alpha01"
    else:
        suffix = f"alpha{str(alpha).replace('.', '')}"
    
    # Initialize pipeline
    # Block discrete uses dual data type
    data_type = 'dual' if model_type in ['dual', 'block_discrete'] else model_type
    pipeline = IntegratedDataPipelineV2(model_type=data_type, random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
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
    
    # Process each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        patient_states = test_data['states'][start_idx:end_idx]
        patient_actions = test_data['actions'][start_idx:end_idx]
        
        patient_q_values = []
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
                else:
                    model_action_arr = agent.select_action(state, epsilon=0.0)
                    model_action = int(model_action_arr[0])
                    
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        action_t = torch.FloatTensor([model_action]).unsqueeze(0).to(agent.device)
                        q1 = agent.q1(state_t, action_t).item()
                        q2 = agent.q2(state_t, action_t).item()
                        q_val = min(q1, q2)
                
                if model_action > 0:
                    vp1_used = True
                patient_vp1_concordance.append(model_action == clinician_action)
                
            else:  # Dual or Block Discrete
                if policy:
                    # Use policy wrapper with persistence
                    model_action = policy.select_action(state, patient_id)
                    q_val = policy.get_q_value(state, model_action)
                else:
                    # Use agent directly
                    model_action = agent.select_action(state)
                    
                    if model_type == 'block_discrete':
                        # For block discrete, need to get Q-value differently
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                            # Convert continuous action to discrete index for Q-value computation
                            action_idx = agent.continuous_to_discrete_action(model_action)
                            action_idx_t = torch.LongTensor([action_idx]).to(agent.device)
                            q1 = agent.q1(state_t, action_idx_t).item()
                            q2 = agent.q2(state_t, action_idx_t).item()
                            q_val = min(q1, q2)
                    else:
                        # Dual mixed model
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                            action_t = torch.FloatTensor(model_action).unsqueeze(0).to(agent.device)
                            q1 = agent.q1(state_t, action_t).item()
                            q2 = agent.q2(state_t, action_t).item()
                            q_val = min(q1, q2)
                
                # VP1 is binary (0 or 1), VP2 is continuous [0, 0.5]
                if model_action[0] > 0:  # VP1 is binary
                    vp1_used = True
                if model_action[1] > 0.01:  # VP2 threshold
                    vp2_used = True
                    
                # VP1 Concordance (binary comparison)
                clinician_vp1 = clinician_action[0] if clinician_action.ndim > 0 else clinician_action
                patient_vp1_concordance.append((model_action[0] > 0) == (clinician_vp1 > 0))
                
                # VP2 Concordance for Block Discrete models only
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
            
            patient_q_values.append(q_val)
            q_values_timestep.append(q_val)
        
        vp1_usage.append(vp1_used)
        if model_type in ['dual', 'block_discrete']:
            vp2_usage.append(vp2_used)
        q_values_patient.append(np.mean(patient_q_values))
        vp1_concordances.append(np.mean(patient_vp1_concordance))
        if model_type == 'block_discrete' and patient_vp2_concordance:
            vp2_concordances.append(np.mean(patient_vp2_concordance))
    
    # Calculate metrics
    results = {
        'vp1_usage': np.mean(vp1_usage) * 100,
        'vp2_usage': np.mean(vp2_usage) * 100 if vp2_usage else None,
        'q_per_timestep': np.mean(q_values_timestep),
        'q_per_patient': np.mean(q_values_patient),
        'vp1_concordance': np.mean(vp1_concordances) * 100,
        'vp2_concordance': np.mean(vp2_concordances) * 100 if vp2_concordances else None
    }
    
    return results


def generate_latex_table():
    """Generate LaTeX table for Table 1"""
    
    print("="*70)
    print(" GENERATING TABLE 1: CQL Model Comparison")
    print("="*70)
    
    # Evaluate models for both alpha values
    results = {}
    
    # Include alpha=0.0 in evaluation
    for alpha in [0.0, 0.001, 0.01]:
        print(f"\nEvaluating models with alpha={alpha}...")
        
        # Binary CQL with persistence
        print(f"  Binary CQL...")
        binary_results = evaluate_model('binary', alpha, apply_persistence=True)
        
        # Dual Mixed CQL
        print(f"  Dual Mixed CQL...")
        dual_results = evaluate_model('dual', alpha, apply_persistence=True)
        
        results[alpha] = {
            'binary': binary_results,
            'dual': dual_results,
            'block_discrete': {}
        }
        
        # Block Discrete CQL with different bin sizes
        for bins in [3, 5, 10]:
            print(f"  Block Discrete CQL (bins={bins})...")
            block_results = evaluate_model('block_discrete', alpha, apply_persistence=True, vp2_bins=bins)
            results[alpha]['block_discrete'][bins] = block_results
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{Comprehensive comparison of CQL models including Binary, Dual Mixed, and Block Discrete (with 3, 5, 10 bins) variants across different conservatism levels ($\alpha$).}
\label{tab:cql_comparison}
\begin{tabular}{llcccccc}
\toprule
Model & Config & $\alpha$ & VP1 (\%) & VP2 (\%) & Q/time & VP1 Conc. (\%) & VP2 Conc. (\%) \\
\midrule
Clinician & -- & -- & 38.8 & 99.8 & 0.000 & 100.0 & -- \\
\midrule"""
    
    # Add results for each alpha
    for alpha in [0.0, 0.001, 0.01]:
        b_res = results[alpha]['binary']
        d_res = results[alpha]['dual']
        
        # Helper function to format values with None handling
        def fmt(val, fmt_str=".1f"):
            if val is None:
                return "N/A"
            try:
                return f"{val:{fmt_str}}"
            except:
                return "N/A"
        
        # Binary CQL
        latex += f"""
Binary CQL & -- & {alpha:.3f} & {fmt(b_res.get('vp1_usage'))} & -- & {fmt(b_res.get('q_per_timestep'), '.3f')} & {fmt(b_res.get('vp1_concordance'))} & -- \\\\"""
        
        # Dual Mixed CQL
        latex += f"""
Dual Mixed & -- & {alpha:.3f} & {fmt(d_res.get('vp1_usage'))} & {fmt(d_res.get('vp2_usage'))} & {fmt(d_res.get('q_per_timestep'), '.3f')} & {fmt(d_res.get('vp1_concordance'))} & -- \\\\"""
        
        # Block Discrete CQL models
        for bins in [3, 5, 10]:
            bd_res = results[alpha]['block_discrete'][bins]
            
            if bd_res:
                latex += f"""
Block Discrete & {bins} bins & {alpha:.3f} & {fmt(bd_res.get('vp1_usage'))} & {fmt(bd_res.get('vp2_usage'))} & {fmt(bd_res.get('q_per_timestep'), '.3f')} & {fmt(bd_res.get('vp1_concordance'))} & {fmt(bd_res.get('vp2_concordance'))} \\\\"""
            else:
                latex += f"""
Block Discrete & {bins} bins & {alpha:.3f} & N/A & N/A & N/A & N/A & N/A \\\\"""
        
        if alpha in [0.0, 0.001]:  # Add separator after each alpha group except the last
            latex += "\n\\midrule"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open('table1_cql_comparison.tex', 'w') as f:
        f.write(latex)
    
    print("\n✓ LaTeX table saved to: table1_cql_comparison.tex")
    
    # Also print results
    print("\nNumerical Results:")
    print("-"*70)
    for alpha in [0.0, 0.001, 0.01]:
        print(f"\nAlpha = {alpha}:")
        b_res = results[alpha]['binary']
        d_res = results[alpha]['dual']
        
        # Safe formatting function
        def safe_fmt(val, suffix='%', decimals=1):
            if val is None:
                return "N/A"
            if decimals == 3:
                return f"{val:.3f}"
            return f"{val:.{decimals}f}{suffix}"
        
        print(f"  Binary CQL:     VP1={safe_fmt(b_res.get('vp1_usage'))}, Q/t={safe_fmt(b_res.get('q_per_timestep'), '', 3)}, VP1-Conc={safe_fmt(b_res.get('vp1_concordance'))}")
        print(f"  Dual Mixed:     VP1={safe_fmt(d_res.get('vp1_usage'))}, VP2={safe_fmt(d_res.get('vp2_usage'))}, Q/t={safe_fmt(d_res.get('q_per_timestep'), '', 3)}, VP1-Conc={safe_fmt(d_res.get('vp1_concordance'))}")
        for bins in [3, 5, 10]:
            bd_res = results[alpha]['block_discrete'][bins]
            if bd_res:
                print(f"  Block Disc({bins:2d}): VP1={safe_fmt(bd_res.get('vp1_usage'))}, VP2={safe_fmt(bd_res.get('vp2_usage'))}, Q/t={safe_fmt(bd_res.get('q_per_timestep'), '', 3)}, VP1-C={safe_fmt(bd_res.get('vp1_concordance'))}, VP2-C={safe_fmt(bd_res.get('vp2_concordance'))}")
            else:
                print(f"  Block Disc({bins:2d}): N/A")
    
    return results


if __name__ == "__main__":
    results = generate_latex_table()
    print("\n✅ Table 1 generation complete!")