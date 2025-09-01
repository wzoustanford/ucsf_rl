#!/usr/bin/env python3
"""
Generate Table 1: Comprehensive comparison of LSTM CQL models
Outputs LaTeX table directly
"""

import numpy as np
import torch
import os
import sys

# Add parent directory to path to import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from medical_sequence_buffer import MedicalSequenceBuffer, SequenceDataLoader
from lstm_block_discrete_cql_network import LSTMBlockDiscreteCQL
from vaso_init_policy import VasopressorInitiationPolicy, DualVasopressorInitiationPolicy


def evaluate_lstm_model(alpha=0.001, vp2_bins=5, sequence_length=20, burn_in_length=8):
    """Evaluate an LSTM model on test set"""
    
    # Initialize pipeline for dual mode (norepinephrine as separate action)
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # State dimension is without norepinephrine
    state_dim = train_data['states'].shape[1]
    
    # Format alpha string for filename
    if alpha == 0.0:
        alpha_str = "0.0000"
    elif alpha < 0.0001:
        alpha_str = f"{alpha:.1e}".replace('.', 'p').replace('-', 'm')
    else:
        alpha_str = f"{alpha:.4f}"
    
    # Load LSTM Block Discrete model
    model_path = f'experiment/lstm_block_discrete_cql_alpha{alpha_str}_bins{vp2_bins}_hdim32_ldim32_best.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    
    # Get bin centers from checkpoint
    vp2_bin_centers = checkpoint.get('vp2_bin_centers', np.linspace(0, 0.5, vp2_bins + 1)[:-1] + 0.025)
    
    # Initialize LSTM agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = LSTMBlockDiscreteCQL(
        state_dim=state_dim,
        num_actions=vp2_bins,
        action_bins=vp2_bin_centers,
        hidden_dim=32,
        lstm_hidden=32,
        num_lstm_layers=2,
        alpha=alpha,
        gamma=0.95,
        tau=0.8,
        lr=1e-3,
        device=device
    )
    
    # Load model weights
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
    agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    
    # Helper function to convert continuous doses to discrete indices
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    def continuous_to_discrete(continuous_doses):
        action_indices = np.digitize(continuous_doses, vp2_bin_edges) - 1
        return np.clip(action_indices, 0, vp2_bins - 1)
    
    # Evaluate using sequences
    agent.q1.eval()
    agent.q2.eval()
    
    vp2_usage = []
    q_values_timestep = []
    q_values_patient = []
    vp2_concordances = []
    delta_q_timestep = []  # Delta Q per timestep
    delta_q_patient = []  # Delta Q per patient
    
    # Process a subset of patients for faster evaluation
    patient_ids = list(pipeline.test_patient_groups.keys())
    max_patients = len(patient_ids)  # Evaluate on first 50 patients
    
    for i, patient_id in enumerate(patient_ids[:max_patients]):
        if i % 10 == 0:
            print(f"    Processing patient {i+1}/{max_patients}...")
        
        start_idx, end_idx = pipeline.test_patient_groups[patient_id]
        patient_length = end_idx - start_idx
        
        # Skip if patient sequence is too short
        if patient_length < sequence_length:
            continue
            
        patient_states = test_data['states'][start_idx:end_idx]
        patient_actions = test_data['actions'][start_idx:end_idx]
        
        patient_q_values = []
        patient_delta_q = []  # Track delta Q for this patient
        patient_vp2_concordance = []
        vp2_used = False
        
        # Process patient in non-overlapping sequences for speed
        for seq_start in range(0, patient_length - sequence_length + 1, sequence_length):
            seq_end = min(seq_start + sequence_length, patient_length)
            
            # Get sequence states and actions
            seq_states = patient_states[seq_start:seq_end]
            seq_actions = patient_actions[seq_start:seq_end]
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(seq_states).unsqueeze(0).to(device)
            
            # Split into burn-in and evaluation
            burn_in_states = states_tensor[:, :burn_in_length, :]
            eval_states = states_tensor[:, burn_in_length:, :]
            
            with torch.no_grad():
                # Initialize hidden state
                hidden1 = agent.q1.init_hidden(1, device)
                hidden2 = agent.q2.init_hidden(1, device)
                
                # Burn-in phase
                if burn_in_states.shape[1] > 0:
                    _, hidden1 = agent.q1.forward(burn_in_states, hidden1)
                    _, hidden2 = agent.q2.forward(burn_in_states, hidden2)
                
                # Get Q-values for all actions
                q1_all, _ = agent.q1.forward(eval_states, hidden1)
                q2_all, _ = agent.q2.forward(eval_states, hidden2)
                
                # Take max Q-value (best action) for each timestep
                q1_max, selected_actions1 = torch.max(q1_all, dim=-1)
                q2_max, selected_actions2 = torch.max(q2_all, dim=-1)
                
                # Use minimum of two Q-networks for optimal action
                q_values = torch.min(q1_max, q2_max)
                selected_actions = selected_actions1  # Use actions from Q1
                
                # Convert selected actions to continuous doses
                selected_actions_np = selected_actions.squeeze(0).cpu().numpy()
                model_doses = vp2_bin_centers[selected_actions_np]
                
                # Compare with clinician actions for concordance
                eval_actions = seq_actions[burn_in_length:]
                for t in range(len(eval_actions)):
                    clinician_vp2 = eval_actions[t, 1] if len(eval_actions[t]) > 1 else 0.0
                    
                    # Check if VP2 was used
                    if model_doses[t] > 0.01:
                        vp2_used = True
                    
                    # Calculate concordance
                    clinician_bin = continuous_to_discrete(np.array([clinician_vp2]))[0]
                    model_bin = selected_actions_np[t]
                    patient_vp2_concordance.append(model_bin == clinician_bin)
                    
                    # Get Q-value for clinician's action
                    q1_clinician = q1_all[0, t, clinician_bin].item()
                    q2_clinician = q2_all[0, t, clinician_bin].item()
                    q_clinician = min(q1_clinician, q2_clinician)
                    
                    # Calculate delta Q (model optimal - clinician)
                    q_optimal = q_values[0, t].item()
                    delta_q = q_optimal - q_clinician
                    
                    # Store values
                    patient_q_values.append(q_optimal)
                    q_values_timestep.append(q_optimal)
                    patient_delta_q.append(delta_q)
                    delta_q_timestep.append(delta_q)
        
        if patient_q_values:  # Only if we processed sequences
            vp2_usage.append(vp2_used)
            q_values_patient.append(np.mean(patient_q_values))
            delta_q_patient.append(np.mean(patient_delta_q))  # Add patient-level delta Q
            if patient_vp2_concordance:
                vp2_concordances.append(np.mean(patient_vp2_concordance))
    
    # Calculate metrics
    results = {
        'vp2_usage': np.mean(vp2_usage) * 100 if vp2_usage else 0.0,
        'q_per_timestep': np.mean(q_values_timestep) if q_values_timestep else 0.0,
        'q_per_patient': np.mean(q_values_patient) if q_values_patient else 0.0,
        'delta_q_per_timestep': np.mean(delta_q_timestep) if delta_q_timestep else 0.0,  # Average delta Q per timestep
        'delta_q_per_patient': np.mean(delta_q_patient) if delta_q_patient else 0.0,  # Average delta Q per patient
        'vp2_concordance': np.mean(vp2_concordances) * 100 if vp2_concordances else 0.0
    }
    
    return results


def generate_latex_table():
    """Generate LaTeX table for LSTM models"""
    
    print("="*70)
    print(" GENERATING TABLE 1: LSTM Block Discrete CQL Model Evaluation")
    print("="*70)
    
    # Read model list from file
    with open('experiment/lstm_32dim.md', 'r') as f: #lstm_eval_200epochs_list.md #lstm_eval_200epochs_list.md
        model_files = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse model configurations from filenames
    model_configs = []
    for model_file in model_files:
        # Parse alpha and bins from filename
        # Format: lstm_block_discrete_cql_alpha{alpha}_bins{bins}_best.pt
        parts = model_file.split('_')
        
        # Find alpha value
        alpha_part = [p for p in parts if p.startswith('alpha')][0]
        alpha_str = alpha_part.replace('alpha', '')
        
        # Convert alpha string to float
        if 'em' in alpha_str:  # Scientific notation like 1p0em05
            # Replace p with . and em with e-
            alpha_str = alpha_str.replace('p', '.')
            alpha_str = alpha_str.replace('em', 'e-')
            alpha = float(alpha_str)
        elif alpha_str == '0.0000':
            alpha = 0.0
        else:
            alpha = float(alpha_str)
        
        # Find bins value
        bins_part = [p for p in parts if p.startswith('bins')][0]
        bins = int(bins_part.replace('bins', ''))
        
        model_configs.append((alpha, bins))
    
    # Sort by alpha then bins
    model_configs.sort(key=lambda x: (x[0], x[1]))
    
    # Evaluate models
    results = {}
    
    for alpha, bins in model_configs:
        print(f"\nEvaluating LSTM model with alpha={alpha:.1e}, bins={bins}...")
        
        lstm_results = evaluate_lstm_model(alpha=alpha, vp2_bins=bins)
        
        if alpha not in results:
            results[alpha] = {}
        results[alpha][bins] = lstm_results
    
    # Generate LaTeX
    latex = r"""\begin{table}[ht]
\centering
\caption{Evaluation of LSTM Block Discrete CQL models with different conservatism levels ($\alpha$) and bin configurations.}
\label{tab:lstm_cql_evaluation}
\begin{tabular}{lcccccc}
\toprule
Model & $\alpha$ & Bins & VP2 Usage (\%) & Q/time & $\Delta$Q & VP2 Conc. (\%) \\
\midrule"""
    
    # Helper function to format values
    def fmt(val, fmt_str=".1f"):
        if val is None or val == 0.0:
            return "--"
        try:
            return f"{val:{fmt_str}}"
        except:
            return "--"
    
    # Add results for each model
    for alpha in sorted(results.keys()):
        for bins in sorted(results[alpha].keys()):
            res = results[alpha][bins]
            
            if res:
                # Format alpha for display
                if alpha == 0.0:
                    alpha_display = "0.0"
                elif alpha < 0.0001:
                    alpha_display = f"{alpha:.1e}"
                else:
                    alpha_display = f"{alpha:.4f}"
                
                latex += f"""
LSTM Block Discrete & {alpha_display} & {bins} & {fmt(res.get('vp2_usage'))} & {fmt(res.get('q_per_timestep'), '.3f')} & {fmt(res.get('delta_q_per_timestep'), '.3f')} & {fmt(res.get('vp2_concordance'))} \\\\"""
            else:
                latex += f"""
LSTM Block Discrete & {alpha:.1e} & {bins} & -- & -- & -- & -- \\\\"""
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open('table1_lstm_v2_alltestpatients_hdim32_best_deltaq_cql_evaluation.tex', 'w') as f:
        f.write(latex)
    
    print("\n✓ LaTeX table saved to: table1_lstm_v2_alltestpatients_hdim32_best_deltaq_cql_evaluation.tex")
    
    # Also print results
    print("\nNumerical Results:")
    print("-"*70)
    
    # Safe formatting function
    def safe_fmt(val, suffix='%', decimals=1):
        if val is None or val == 0.0:
            return "N/A"
        if decimals == 3:
            return f"{val:.3f}"
        return f"{val:.{decimals}f}{suffix}"
    
    for alpha in sorted(results.keys()):
        # Format alpha for display
        if alpha == 0.0:
            alpha_display = "0.0"
        elif alpha < 0.0001:
            alpha_display = f"{alpha:.1e}"
        else:
            alpha_display = f"{alpha:.4f}"
        
        print(f"\nAlpha = {alpha_display}:")
        for bins in sorted(results[alpha].keys()):
            res = results[alpha][bins]
            if res:
                print(f"  LSTM({bins:2d} bins): VP2={safe_fmt(res.get('vp2_usage'))}, Q/t={safe_fmt(res.get('q_per_timestep'), '', 3)}, ΔQ={safe_fmt(res.get('delta_q_per_timestep'), '', 3)}, VP2-Conc={safe_fmt(res.get('vp2_concordance'))}")
            else:
                print(f"  LSTM({bins:2d} bins): N/A")
    
    return results


if __name__ == "__main__":
    results = generate_latex_table()
    print("\n✅ Table 1 generation complete!")