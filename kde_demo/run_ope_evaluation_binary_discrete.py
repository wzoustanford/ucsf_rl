"""
Run off-policy evaluation for Binary and Block Discrete CQL models
Generates comprehensive comparison table in LaTeX format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kde_demo.q_softmax_block_discrete_cql_ope import (
    evaluate_binary_cql, 
    evaluate_block_discrete_cql
)


def main():
    print("\n" + "="*80)
    print("OFF-POLICY EVALUATION: BINARY vs BLOCK DISCRETE CQL")
    print("="*80)
    print("\nEvaluating three models on the same test set:")
    print("1. Binary CQL (VP1 only, α=0.001)")
    print("2. Block Discrete CQL (VP1+VP2, α=0.0, 10 bins) - Block KDE")
    print("3. Block Discrete CQL (VP1+VP2, α=0.0, 10 bins) - Q-Softmax")
    
    results = {}
    
    # 1. Binary CQL
    print("\n\n" + "="*80)
    print("MODEL 1: BINARY CQL")
    print("="*80)
    binary_results = evaluate_binary_cql(
        model_path='experiment/binary_cql_unified_alpha001_best.pt',
        method='binary_kde'
    )
    results['Binary CQL'] = binary_results
    
    # 2. Block Discrete CQL with Block KDE
    print("\n\n" + "="*80)
    print("MODEL 2: BLOCK DISCRETE CQL (Block KDE)")
    print("="*80)
    block_kde_results = evaluate_block_discrete_cql(
        model_path='experiment/block_discrete_cql_alpha0.0000_bins10_best.pt',
        vp2_bins=10,
        method='block_kde'
    )
    results['Block Discrete (KDE)'] = block_kde_results
    
    # 3. Block Discrete CQL with Q-Softmax
    print("\n\n" + "="*80)
    print("MODEL 3: BLOCK DISCRETE CQL (Q-Softmax)")
    print("="*80)
    qsoftmax_results = evaluate_block_discrete_cql(
        model_path='experiment/block_discrete_cql_alpha0.0000_bins10_best.pt',
        vp2_bins=10,
        method='q_softmax'
    )
    results['Block Discrete (Q-Softmax)'] = qsoftmax_results
    
    # Generate LaTeX table
    print("\n\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Off-Policy Evaluation Results for CQL Models on Test Set}
\label{tab:cql_ope_results}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{Model} & \multirow{2}{*}{Method} & \multicolumn{2}{c}{Policy Value Estimate} & \multicolumn{2}{c}{95\% CI} & \multirow{2}{*}{ESS (\%)} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
 & & IS & WIS & IS & WIS & \\
\midrule
"""
    
    # Add behavioral policy row
    behavioral_reward = results['Binary CQL']['behavioral_reward']
    latex_table += f"Behavioral (Clinician) & - & \\multicolumn{{2}}{{c}}{{{behavioral_reward:.4f}}} & \\multicolumn{{2}}{{c}}{{-}} & - \\\\\n"
    latex_table += r"\midrule" + "\n"
    
    # Add each model's results
    model_configs = [
        ('Binary CQL ($\\alpha$=0.001)', 'KDE', 'Binary CQL'),
        ('Block Discrete ($\\alpha$=0)', 'Block KDE', 'Block Discrete (KDE)'),
        ('Block Discrete ($\\alpha$=0)', 'Q-Softmax', 'Block Discrete (Q-Softmax)')
    ]
    
    for model_name, method_name, result_key in model_configs:
        res = results[result_key]
        is_est = res['is_estimate']
        wis_est = res['wis_estimate']
        is_ci_lower, is_ci_upper = res['is_ci']
        wis_ci_lower, wis_ci_upper = res['wis_ci']
        ess_pct = res['ess_ratio'] * 100
        
        latex_table += f"{model_name} & {method_name} & "
        latex_table += f"{is_est:.4f} & {wis_est:.4f} & "
        latex_table += f"[{is_ci_lower:.3f}, {is_ci_upper:.3f}] & "
        latex_table += f"[{wis_ci_lower:.3f}, {wis_ci_upper:.3f}] & "
        latex_table += f"{ess_pct:.1f} \\\\\n"
    
    latex_table += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: IS = Importance Sampling, WIS = Weighted Importance Sampling, ESS = Effective Sample Size.
\item Binary CQL uses VP1 only (binary vasopressor). Block Discrete uses VP1 (binary) + VP2 (10 discrete bins).
\item All models evaluated on the same test set of 31,289 transitions from 595 patients.
\item Positive values indicate improvement over behavioral policy.
\end{tablenotes}
\end{table}
"""
    
    print(latex_table)
    
    # Save to file
    with open('kde_demo/ope_comparison_binary_discrete.tex', 'w') as f:
        f.write(latex_table)
    print("\nTable saved to: kde_demo/ope_comparison_binary_discrete.tex")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBehavioral Policy Average Reward: {behavioral_reward:.4f}")
    print("\nModel Performance (WIS Estimates):")
    for model_name, _, result_key in model_configs:
        wis = results[result_key]['wis_estimate']
        improvement = wis - behavioral_reward
        pct_improvement = (improvement / abs(behavioral_reward)) * 100 if behavioral_reward != 0 else 0
        print(f"  {model_name:30s}: {wis:7.4f} (Δ={improvement:+.4f}, {pct_improvement:+.1f}%)")


if __name__ == "__main__":
    main()