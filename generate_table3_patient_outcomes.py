#!/usr/bin/env python3
"""
Generate Table 3: Patient outcome analysis and Q-value improvements
Outputs LaTeX table directly
"""

import numpy as np
import torch
import os
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from train_binary_cql import BinaryCQL
from train_cql_stable import StableCQL
from binary_cql_evaluation import VasopressorInitiationPolicy


def evaluate_by_outcome(model_type='binary', alpha=0.001, apply_persistence=False):
    """Evaluate model performance stratified by patient outcome"""
    
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
        model_path = f'experiment/binary_cql_unified_{suffix}_best.pt'
        if not os.path.exists(model_path):
            model_path = f'experiment/binary_cql_unified_{suffix}_final.pt'
        
        agent = BinaryCQL(state_dim=18, alpha=alpha, tau=0.8, lr=1e-3)
        agent.load(model_path)
        
        if apply_persistence:
            policy = VasopressorInitiationPolicy(agent)
        else:
            policy = None
    else:
        model_path = f'experiment/dual_cql_unified_{suffix}_best.pt'
        if not os.path.exists(model_path):
            model_path = f'experiment/dual_cql_unified_{suffix}_final.pt'
        
        checkpoint = torch.load(model_path, map_location='cuda')
        agent = StableCQL(state_dim=17, action_dim=2, alpha=alpha, tau=0.8, lr=1e-3)
        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        policy = None
    
    # Evaluate on test data
    patient_groups = test_data['patient_groups']
    
    # Separate tracking for survived/died
    survived_vp1 = []
    survived_q_values = []
    died_vp1 = []
    died_q_values = []
    all_q_values = []
    
    # Process each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        patient_states = test_data['states'][start_idx:end_idx]
        patient_rewards = test_data['rewards'][start_idx:end_idx]
        patient_dones = test_data['dones'][start_idx:end_idx]
        
        # Check mortality
        is_died = patient_rewards[-1] < 0 and patient_dones[-1] == 1.0
        
        patient_q_values = []
        vp1_used = False
        
        if policy and model_type == 'binary':
            policy.reset()
        
        for t in range(len(patient_states)):
            state = patient_states[t]
            
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
                    
            else:  # Dual
                model_action = agent.select_action(state, epsilon=0.0)
                
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    action_t = torch.FloatTensor(model_action).unsqueeze(0).to(agent.device)
                    q1 = agent.q1(state_t, action_t).item()
                    q2 = agent.q2(state_t, action_t).item()
                    q_val = min(q1, q2)
                
                if model_action[0] > 0.5:
                    vp1_used = True
            
            patient_q_values.append(q_val)
            all_q_values.append(q_val)
        
        # Store results by outcome
        patient_avg_q = np.mean(patient_q_values)
        
        if is_died:
            died_vp1.append(vp1_used)
            died_q_values.append(patient_avg_q)
        else:
            survived_vp1.append(vp1_used)
            survived_q_values.append(patient_avg_q)
    
    # Calculate metrics
    results = {
        'overall_q': np.mean(all_q_values),
        'survived_vp1_pct': np.mean(survived_vp1) * 100 if survived_vp1 else 0,
        'survived_q': np.mean(survived_q_values) if survived_q_values else 0,
        'died_vp1_pct': np.mean(died_vp1) * 100 if died_vp1 else 0,
        'died_q': np.mean(died_q_values) if died_q_values else 0
    }
    
    return results


def generate_latex_table():
    """Generate LaTeX table for Table 3"""
    
    print("="*70)
    print(" GENERATING TABLE 3: Patient Outcome Analysis")
    print("="*70)
    
    # Evaluate models for both alpha values
    results = {}
    
    for alpha in [0.001, 0.01]:
        print(f"\nEvaluating models with alpha={alpha}...")
        
        # Binary CQL with persistence
        print(f"  Binary CQL...")
        binary_results = evaluate_by_outcome('binary', alpha, apply_persistence=True)
        
        # Dual CQL
        print(f"  Dual CQL...")
        dual_results = evaluate_by_outcome('dual', alpha)
        
        results[alpha] = {
            'binary': binary_results,
            'dual': dual_results
        }
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{Patient outcome analysis showing VP1 usage and Q-values stratified by survival status. Higher Q-values in survived patients indicate better expected outcomes from the learned policies.}
\label{tab:patient_outcomes}
\begin{tabular}{lcccccc}
\toprule
Model & $\alpha$ & Overall Q/time & Survived VP1 (\%) & Survived Q & Died VP1 (\%) & Died Q \\
\midrule
Clinician & -- & 0.000 & 30.2 & 0.000 & 55.4 & 0.000 \\
\midrule"""
    
    # Add results for each alpha
    for alpha in [0.001, 0.01]:
        b_res = results[alpha]['binary']
        d_res = results[alpha]['dual']
        
        latex += f"""
Binary CQL & {alpha:.3f} & {b_res['overall_q']:.3f} & {b_res['survived_vp1_pct']:.1f} & {b_res['survived_q']:.3f} & {b_res['died_vp1_pct']:.1f} & {b_res['died_q']:.3f} \\\\
Dual CQL & {alpha:.3f} & {d_res['overall_q']:.3f} & {d_res['survived_vp1_pct']:.1f} & {d_res['survived_q']:.3f} & {d_res['died_vp1_pct']:.1f} & {d_res['died_q']:.3f} \\\\"""
        
        if alpha == 0.001:
            latex += "\n\\midrule"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open('table3_patient_outcomes.tex', 'w') as f:
        f.write(latex)
    
    print("\n✓ LaTeX table saved to: table3_patient_outcomes.tex")
    
    # Also print results
    print("\nNumerical Results:")
    print("-"*70)
    for alpha in [0.001, 0.01]:
        print(f"\nAlpha = {alpha}:")
        b_res = results[alpha]['binary']
        d_res = results[alpha]['dual']
        print(f"  Binary CQL: Overall Q={b_res['overall_q']:.3f}, Survived VP1={b_res['survived_vp1_pct']:.1f}%, Died VP1={b_res['died_vp1_pct']:.1f}%")
        print(f"  Dual CQL:   Overall Q={d_res['overall_q']:.3f}, Survived VP1={d_res['survived_vp1_pct']:.1f}%, Died VP1={d_res['died_vp1_pct']:.1f}%")
    
    return results


if __name__ == "__main__":
    results = generate_latex_table()
    print("\n✅ Table 3 generation complete!")