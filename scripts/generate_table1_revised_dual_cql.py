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
from vaso_init_policy import VasopressorInitiationPolicy, DualVasopressorInitiationPolicy


def evaluate_model(model_type='binary', alpha=0.001, apply_persistence=False):
    """Evaluate a model on test set"""
    
    # Map alpha to file suffix
    if alpha == 0.001:
        suffix = "alpha001"
    elif alpha == 0.01:
        suffix = "alpha01"
    else:
        suffix = f"alpha{str(alpha).replace('.', '')}"
    
    # Initialize pipeline
    pipeline = IntegratedDataPipelineV2(model_type=model_type, random_seed=42)
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
    concordances = []
    
    # Process each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        patient_states = test_data['states'][start_idx:end_idx]
        patient_actions = test_data['actions'][start_idx:end_idx]
        
        patient_q_values = []
        patient_concordance = []
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
                patient_concordance.append(model_action == clinician_action)
                
            else:  # Dual
                if policy:
                    # Use policy wrapper with persistence
                    model_action = policy.select_action(state, patient_id)
                    q_val = policy.get_q_value(state, model_action)
                else:
                    # Use agent directly
                    model_action = agent.select_action(state)
                    
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
                    
                # Concordance for VP1 (binary comparison)
                clinician_vp1 = clinician_action[0] if clinician_action.ndim > 0 else clinician_action
                patient_concordance.append((model_action[0] > 0) == (clinician_vp1 > 0))
            
            patient_q_values.append(q_val)
            q_values_timestep.append(q_val)
        
        vp1_usage.append(vp1_used)
        if model_type == 'dual':
            vp2_usage.append(vp2_used)
        q_values_patient.append(np.mean(patient_q_values))
        concordances.append(np.mean(patient_concordance))
    
    # Calculate metrics
    results = {
        'vp1_usage': np.mean(vp1_usage) * 100,
        'vp2_usage': np.mean(vp2_usage) * 100 if vp2_usage else None,
        'q_per_timestep': np.mean(q_values_timestep),
        'q_per_patient': np.mean(q_values_patient),
        'concordance': np.mean(concordances) * 100
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
        
        # Dual CQL
        print(f"  Dual CQL...")
        dual_results = evaluate_model('dual', alpha, apply_persistence=True)
        
        results[alpha] = {
            'binary': binary_results,
            'dual': dual_results
        }
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{Comprehensive comparison of Conservative Q-Learning (CQL) models with different conservatism levels ($\alpha$) for vasopressor control. VP1 and VP2 refer to the first and second vasopressor agents respectively. Q-values represent expected returns, concordance measures agreement with clinical decisions, and Q-ratio compares model Q-values to baseline Binary CQL.}
\label{tab:cql_comparison}
\begin{tabular}{lccccccc}
\toprule
Model & $\alpha$ & VP1 (\%) & VP2 (\%) & Q/time & Q/patient & Concord (\%) & Q-Ratio \\
\midrule
Clinician & -- & 38.8 & 99.8 & 0.000 & 0.000 & 100.0 & -- \\
\midrule"""
    
    # Add results for each alpha
    for alpha in [0.0, 0.001, 0.01]:
        b_res = results[alpha]['binary']
        d_res = results[alpha]['dual']
        
        # Calculate Q-ratio (dual/binary)
        q_ratio = d_res['q_per_timestep'] / b_res['q_per_timestep'] if b_res['q_per_timestep'] != 0 else 0
        
        latex += f"""
Binary CQL & {alpha:.3f} & {b_res['vp1_usage']:.1f} & -- & {b_res['q_per_timestep']:.3f} & {b_res['q_per_patient']:.3f} & {b_res['concordance']:.1f} & 1.0$\\times$ \\\\
Dual CQL & {alpha:.3f} & {d_res['vp1_usage']:.1f} & {d_res['vp2_usage']:.1f} & {d_res['q_per_timestep']:.3f} & {d_res['q_per_patient']:.3f} & {d_res['concordance']:.1f} & {q_ratio:.1f}$\\times$ \\\\"""
        
        if alpha in [0.0, 0.001]:  # Add separator after each alpha pair except the last
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
        print(f"  Binary CQL: VP1={b_res['vp1_usage']:.1f}%, Q/t={b_res['q_per_timestep']:.3f}, Concord={b_res['concordance']:.1f}%")
        print(f"  Dual CQL:   VP1={d_res['vp1_usage']:.1f}%, VP2={d_res['vp2_usage']:.1f}%, Q/t={d_res['q_per_timestep']:.3f}, Concord={d_res['concordance']:.1f}%")
    
    return results


if __name__ == "__main__":
    results = generate_latex_table()
    print("\n✅ Table 1 generation complete!")