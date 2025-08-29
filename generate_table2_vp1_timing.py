#!/usr/bin/env python3
"""
Generate Table 2: VP1 initiation timing distribution
Outputs LaTeX table directly
"""

import numpy as np
import torch
import os
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from train_binary_cql import BinaryCQL
from train_cql_stable import StableCQL
from binary_cql_evaluation import VasopressorInitiationPolicy


def evaluate_vp1_timing(model_type='binary', alpha=0.001, apply_persistence=False):
    """Evaluate VP1 initiation timing for a model"""
    
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
    
    vp1_initiation_times = []
    vp1_usage_count = 0
    
    # Process each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        patient_states = test_data['states'][start_idx:end_idx]
        
        vp1_initiated = False
        vp1_init_time = None
        
        if policy and model_type == 'binary':
            policy.reset()
        
        for t in range(len(patient_states)):
            state = patient_states[t]
            
            if model_type == 'binary':
                if policy:
                    model_action = policy.select_action(state, patient_id, epsilon=0.0)
                else:
                    model_action_arr = agent.select_action(state, epsilon=0.0)
                    model_action = int(model_action_arr[0])
                
                if model_action > 0 and not vp1_initiated:
                    vp1_initiated = True
                    vp1_init_time = t
                    
            else:  # Dual
                model_action = agent.select_action(state, epsilon=0.0)
                
                if model_action[0] > 0.5 and not vp1_initiated:
                    vp1_initiated = True
                    vp1_init_time = t
        
        if vp1_initiated:
            vp1_usage_count += 1
            vp1_initiation_times.append(vp1_init_time)
    
    # Analyze timing distribution
    vp1_usage_pct = (vp1_usage_count / len(patient_groups)) * 100
    
    if vp1_initiation_times:
        times = np.array(vp1_initiation_times)
        total = len(times)
        
        timing_dist = {
            'vp1_usage_pct': vp1_usage_pct,
            '0-6h': np.sum(times <= 6),
            '6-12h': np.sum((times > 6) & (times <= 12)),
            '12-24h': np.sum((times > 12) & (times <= 24)),
            '24-48h': np.sum((times > 24) & (times <= 48)),
            '>48h': np.sum(times > 48),
            '<=6h_pct': 100 * np.sum(times <= 6) / total,
            '<=12h_pct': 100 * np.sum(times <= 12) / total,
            '<=24h_pct': 100 * np.sum(times <= 24) / total,
            '<=48h_pct': 100 * np.sum(times <= 48) / total,
            'median': np.median(times)
        }
    else:
        timing_dist = {
            'vp1_usage_pct': vp1_usage_pct,
            '0-6h': 0, '6-12h': 0, '12-24h': 0, '24-48h': 0, '>48h': 0,
            '<=6h_pct': 0, '<=12h_pct': 0, '<=24h_pct': 0, '<=48h_pct': 0,
            'median': 0
        }
    
    return timing_dist


def generate_latex_table():
    """Generate LaTeX table for Table 2"""
    
    print("="*70)
    print(" GENERATING TABLE 2: VP1 Initiation Timing")
    print("="*70)
    
    # Evaluate models for both alpha values
    results = {}
    
    for alpha in [0.001, 0.01]:
        print(f"\nEvaluating models with alpha={alpha}...")
        
        # Binary CQL with persistence
        print(f"  Binary CQL...")
        binary_timing = evaluate_vp1_timing('binary', alpha, apply_persistence=True)
        
        # Dual CQL
        print(f"  Dual CQL...")
        dual_timing = evaluate_vp1_timing('dual', alpha)
        
        results[alpha] = {
            'binary': binary_timing,
            'dual': dual_timing
        }
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{VP1 initiation timing distribution across patient trajectories. Time windows represent hours from ICU admission. The table shows both absolute counts and cumulative percentages for different time thresholds.}
\label{tab:vp1_timing}
\begin{tabular}{lcccccccccc}
\toprule
Model & $\alpha$ & VP1 (\%) & 0-6h & 6-12h & 12-24h & 24-48h & >48h & $\leq$6h (\%) & $\leq$24h (\%) & Median (h) \\
\midrule
Clinician & -- & 37.1 & 147 & 35 & 25 & 12 & 2 & 66.5 & 93.7 & 2.0 \\
\midrule"""
    
    # Add results for each alpha
    for alpha in [0.001, 0.01]:
        b_tim = results[alpha]['binary']
        d_tim = results[alpha]['dual']
        
        latex += f"""
Binary CQL & {alpha:.3f} & {b_tim['vp1_usage_pct']:.1f} & {b_tim['0-6h']} & {b_tim['6-12h']} & {b_tim['12-24h']} & {b_tim['24-48h']} & {b_tim['>48h']} & {b_tim['<=6h_pct']:.1f} & {b_tim['<=24h_pct']:.1f} & {b_tim['median']:.1f} \\\\
Dual CQL & {alpha:.3f} & {d_tim['vp1_usage_pct']:.1f} & {d_tim['0-6h']} & {d_tim['6-12h']} & {d_tim['12-24h']} & {d_tim['24-48h']} & {d_tim['>48h']} & {d_tim['<=6h_pct']:.1f} & {d_tim['<=24h_pct']:.1f} & {d_tim['median']:.1f} \\\\"""
        
        if alpha == 0.001:
            latex += "\n\\midrule"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open('table2_vp1_timing.tex', 'w') as f:
        f.write(latex)
    
    print("\n✓ LaTeX table saved to: table2_vp1_timing.tex")
    
    # Also print results
    print("\nNumerical Results:")
    print("-"*70)
    for alpha in [0.001, 0.01]:
        print(f"\nAlpha = {alpha}:")
        b_tim = results[alpha]['binary']
        d_tim = results[alpha]['dual']
        print(f"  Binary CQL: VP1={b_tim['vp1_usage_pct']:.1f}%, ≤6h={b_tim['<=6h_pct']:.1f}%, Median={b_tim['median']:.1f}h")
        print(f"  Dual CQL:   VP1={d_tim['vp1_usage_pct']:.1f}%, ≤6h={d_tim['<=6h_pct']:.1f}%, Median={d_tim['median']:.1f}h")
    
    return results


if __name__ == "__main__":
    results = generate_latex_table()
    print("\n✅ Table 2 generation complete!")