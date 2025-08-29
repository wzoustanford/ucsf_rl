#!/usr/bin/env python3
"""
Quick test of VP2 concordance calculation for block discrete models
"""

import numpy as np
import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from run_block_discrete_cql_allalphas import DualBlockDiscreteCQL
from vaso_init_policy import DualVasopressorInitiationPolicy


def test_vp2_concordance(alpha=0.001, vp2_bins=5):
    """Test VP2 concordance calculation"""
    
    print(f"\n{'='*60}")
    print(f"Testing VP2 Concordance for Block Discrete CQL")
    print(f"Alpha={alpha}, Bins={vp2_bins}")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Load model
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
    
    policy = DualVasopressorInitiationPolicy(agent)
    
    # Test on a few patients
    patient_groups = test_data['patient_groups']
    patient_ids = list(patient_groups.keys())[:5]  # Test first 5 patients
    
    # VP2 bin edges
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    
    print(f"VP2 Bin Configuration ({vp2_bins} bins):")
    for i in range(vp2_bins):
        center = (vp2_bin_edges[i] + vp2_bin_edges[i+1]) / 2
        print(f"  Bin {i}: [{vp2_bin_edges[i]:.3f}, {vp2_bin_edges[i+1]:.3f}) → center={center:.3f}")
    
    all_vp1_concordances = []
    all_vp2_concordances = []
    
    for patient_id in patient_ids:
        start_idx, end_idx = patient_groups[patient_id]
        patient_states = test_data['states'][start_idx:end_idx]
        patient_actions = test_data['actions'][start_idx:end_idx]
        
        policy.reset()
        
        patient_vp1_concordance = []
        patient_vp2_concordance = []
        
        print(f"\nPatient {patient_id}: {len(patient_states)} timesteps")
        
        for t in range(min(5, len(patient_states))):  # Show first 5 timesteps
            state = patient_states[t]
            clinician_action = patient_actions[t]
            
            # Get model action
            model_action = policy.select_action(state, patient_id)
            
            # VP1 Concordance
            clinician_vp1 = clinician_action[0]
            model_vp1 = model_action[0]
            vp1_match = (model_vp1 > 0) == (clinician_vp1 > 0)
            patient_vp1_concordance.append(vp1_match)
            
            # VP2 Concordance
            clinician_vp2 = clinician_action[1] if len(clinician_action) > 1 else 0.0
            model_vp2 = model_action[1]
            
            # Find which bin each falls into
            clinician_vp2_bin = np.clip(np.digitize(clinician_vp2, vp2_bin_edges) - 1, 0, vp2_bins - 1)
            model_vp2_bin = np.clip(np.digitize(model_vp2, vp2_bin_edges) - 1, 0, vp2_bins - 1)
            
            vp2_match = (model_vp2_bin == clinician_vp2_bin)
            patient_vp2_concordance.append(vp2_match)
            
            if t < 3:  # Show details for first 3 timesteps
                print(f"  t={t}:")
                print(f"    VP1: Model={model_vp1:.0f}, Clinician={clinician_vp1:.0f} → {'✓' if vp1_match else '✗'}")
                print(f"    VP2: Model={model_vp2:.3f} (bin {model_vp2_bin}), Clinician={clinician_vp2:.3f} (bin {clinician_vp2_bin}) → {'✓' if vp2_match else '✗'}")
        
        vp1_conc = np.mean(patient_vp1_concordance) * 100
        vp2_conc = np.mean(patient_vp2_concordance) * 100
        all_vp1_concordances.append(vp1_conc)
        all_vp2_concordances.append(vp2_conc)
        
        print(f"  Patient Summary: VP1 Concordance={vp1_conc:.1f}%, VP2 Concordance={vp2_conc:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Overall Results (first {len(patient_ids)} patients):")
    print(f"  Mean VP1 Concordance: {np.mean(all_vp1_concordances):.1f}%")
    print(f"  Mean VP2 Concordance: {np.mean(all_vp2_concordances):.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_vp2_concordance(alpha=0.001, vp2_bins=5)