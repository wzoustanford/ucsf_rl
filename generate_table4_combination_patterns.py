#!/usr/bin/env python3
"""
Generate Table 4: Vasopressor combination treatment patterns (Dual CQL only)
Outputs LaTeX table directly
"""

import numpy as np
import torch
import os
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from train_cql_stable import StableCQL


def evaluate_combination_patterns(alpha=0.001):
    """Evaluate vasopressor combination patterns for Dual CQL"""
    
    # Map alpha to file suffix
    if alpha == 0.001:
        suffix = "alpha001"
    elif alpha == 0.01:
        suffix = "alpha01"
    else:
        suffix = f"alpha{str(alpha).replace('.', '')}"
    
    # Initialize pipeline
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Load model
    model_path = f'experiment/dual_cql_unified_{suffix}_best.pt'
    if not os.path.exists(model_path):
        model_path = f'experiment/dual_cql_unified_{suffix}_final.pt'
    
    checkpoint = torch.load(model_path, map_location='cuda')
    agent = StableCQL(state_dim=17, action_dim=2, alpha=alpha, tau=0.8, lr=1e-3)
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    
    # Evaluate on test data
    patient_groups = test_data['patient_groups']
    
    # Count combination patterns
    pattern_counts = {
        'neither': 0,
        'vp1_only': 0,
        'vp2_only': 0,
        'both': 0
    }
    
    # Process each patient
    for patient_id, (start_idx, end_idx) in patient_groups.items():
        patient_states = test_data['states'][start_idx:end_idx]
        
        for t in range(len(patient_states)):
            state = patient_states[t]
            
            # Get model action
            model_action = agent.select_action(state, epsilon=0.0)
            
            # Determine pattern with clinically meaningful thresholds
            # VP1: considered "on" if > 0.05 (5x larger threshold)
            # VP2: considered "on" if > 0.005 mcg/kg/min (5x larger threshold)
            vp1_on = model_action[0] > 0.05
            vp2_on = model_action[1] > 0.005
            
            if not vp1_on and not vp2_on:
                pattern_counts['neither'] += 1
            elif vp1_on and not vp2_on:
                pattern_counts['vp1_only'] += 1
            elif not vp1_on and vp2_on:
                pattern_counts['vp2_only'] += 1
            else:
                pattern_counts['both'] += 1
    
    # Calculate percentages
    total = sum(pattern_counts.values())
    pattern_pcts = {k: (100 * v / total) if total > 0 else 0 for k, v in pattern_counts.items()}
    
    return pattern_pcts


def generate_latex_table():
    """Generate LaTeX table for Table 4"""
    
    print("="*70)
    print(" GENERATING TABLE 4: Combination Treatment Patterns")
    print("="*70)
    
    # Evaluate models for both alpha values
    results = {}
    
    for alpha in [0.001, 0.01]:
        print(f"\nEvaluating Dual CQL with alpha={alpha}...")
        pattern_pcts = evaluate_combination_patterns(alpha)
        results[alpha] = pattern_pcts
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{Vasopressor combination treatment patterns from Dual CQL models. Shows the percentage of timesteps with different vasopressor combinations. Binary CQL only controls VP1, so combination patterns are not applicable.}
\label{tab:combination_patterns}
\begin{tabular}{lcccccc}
\toprule
Model & $\alpha$ & Neither (\%) & VP1 Only (\%) & VP2 Only (\%) & Both VP1+VP2 (\%) \\
\midrule
Clinical Practice & -- & 0.2 & 0.0 & 62.7 & 37.1 \\
\midrule"""
    
    # Add results for each alpha
    for i, alpha in enumerate([0.001, 0.01]):
        pcts = results[alpha]
        latex += f"""
Dual CQL & {alpha:.3f} & {pcts['neither']:.1f} & {pcts['vp1_only']:.1f} & {pcts['vp2_only']:.1f} & {pcts['both']:.1f} \\\\"""
        
        if i == 0:  # Add separator after first alpha
            latex += "\n\\midrule"
    
    # Add note about Binary CQL
    latex += r"""
\midrule
Binary CQL & 0.001 & \multicolumn{4}{c}{\textit{N/A - Only controls VP1}} \\
Binary CQL & 0.010 & \multicolumn{4}{c}{\textit{N/A - Only controls VP1}} \\"""
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open('table4_combination_patterns.tex', 'w') as f:
        f.write(latex)
    
    print("\n✓ LaTeX table saved to: table4_combination_patterns.tex")
    
    # Also print results
    print("\nNumerical Results:")
    print("-"*70)
    for alpha in [0.001, 0.01]:
        pcts = results[alpha]
        print(f"\nDual CQL (alpha={alpha}):")
        print(f"  Neither: {pcts['neither']:.1f}%")
        print(f"  VP1 Only: {pcts['vp1_only']:.1f}%")
        print(f"  VP2 Only: {pcts['vp2_only']:.1f}%")
        print(f"  Both: {pcts['both']:.1f}%")
    
    return results


if __name__ == "__main__":
    results = generate_latex_table()
    print("\n✅ Table 4 generation complete!")