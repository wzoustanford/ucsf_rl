#!/usr/bin/env python3
"""
Unit test to verify CQL loss computation in stepwise model
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_unified_stepwise_cql_allalphas import StepwiseActionSpace, StepwiseQNetwork, UnifiedStepwiseCQL

def test_qnetwork_forward():
    """Test Q-network forward pass with different action formats"""
    print("Testing Q-network forward pass...")
    
    # Setup
    n_base_features = 17
    action_space = StepwiseActionSpace(max_step=0.1)
    state_dim = n_base_features + action_space.n_vp2_bins
    
    q_net = StepwiseQNetwork(state_dim=state_dim, n_actions=action_space.n_actions)
    
    batch_size = 4
    states = torch.randn(batch_size, state_dim)
    
    # Test with action indices
    action_indices = torch.randint(0, action_space.n_actions, (batch_size,))
    q_values_1 = q_net(states, action_indices)
    assert q_values_1.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {q_values_1.shape}"
    print(f"  ✓ Action indices: Q-values shape {q_values_1.shape}")
    
    # Test with one-hot actions
    action_one_hot = F.one_hot(action_indices, num_classes=action_space.n_actions).float()
    q_values_2 = q_net(states, action_one_hot)
    assert q_values_2.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {q_values_2.shape}"
    print(f"  ✓ One-hot actions: Q-values shape {q_values_2.shape}")
    
    # Values should be the same
    assert torch.allclose(q_values_1, q_values_2), "Q-values should be identical for indices vs one-hot"
    print(f"  ✓ Q-values match for both input formats")
    
    return True


def test_cql_loss_computation():
    """Test CQL loss computation step by step"""
    print("\nTesting CQL loss computation...")
    
    # Setup
    n_base_features = 17
    action_space = StepwiseActionSpace(max_step=0.1)
    state_dim = n_base_features + action_space.n_vp2_bins
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = UnifiedStepwiseCQL(
        state_dim=state_dim,
        max_step=0.1,
        alpha=0.001,
        gamma=0.95,
        tau=0.8,
        lr=1e-3,
        device=device
    )
    
    batch_size = 8
    
    # Create synthetic batch on correct device
    states = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randint(0, action_space.n_actions, (batch_size,), device=device)
    rewards = torch.randn(batch_size, device=device)
    next_states = torch.randn(batch_size, state_dim, device=device)
    dones = torch.zeros(batch_size, device=device)
    
    # Extract VP2 doses for masking
    current_vp2_doses = torch.rand(batch_size, device=device) * 0.3  # Random VP2 in [0, 0.3]
    next_vp2_doses = torch.rand(batch_size, device=device) * 0.3
    
    print(f"  Batch size: {batch_size}")
    print(f"  State dim: {state_dim}")
    print(f"  N actions: {action_space.n_actions}")
    
    # Test expanding states for all actions
    states_exp = states.unsqueeze(1).expand(-1, action_space.n_actions, -1)
    states_flat = states_exp.reshape(-1, states.shape[-1])
    print(f"  ✓ Expanded states shape: {states_flat.shape} (expected {(batch_size * action_space.n_actions, state_dim)})")
    
    # Test action indices for all actions
    action_indices = torch.arange(action_space.n_actions, device=states.device)
    action_indices = action_indices.unsqueeze(0).expand(batch_size, -1)
    action_flat = action_indices.reshape(-1)
    print(f"  ✓ Action indices shape: {action_flat.shape} (expected {(batch_size * action_space.n_actions,)})")
    
    # Test Q-value computation for all actions
    all_q = agent.q1(states_flat, action_flat).reshape(batch_size, action_space.n_actions)
    print(f"  ✓ All Q-values shape: {all_q.shape}")
    
    # Test valid action masking
    valid_masks = action_space.get_valid_actions_batch(current_vp2_doses)
    print(f"  ✓ Valid masks shape: {valid_masks.shape}")
    print(f"    Valid actions per sample: {valid_masks.sum(dim=1).tolist()}")
    
    # Apply masking
    masked_q = all_q.clone()
    masked_q[~valid_masks] = -1e8
    
    # Test logsumexp
    temperature = 10.0
    logsumexp_q = torch.logsumexp(masked_q / temperature, dim=1) * temperature
    print(f"  ✓ Logsumexp shape: {logsumexp_q.shape}")
    
    # Get current Q-values for actual actions
    current_q = agent.q1(states, actions).squeeze()
    print(f"  ✓ Current Q shape: {current_q.shape}")
    
    # Compute CQL loss
    cql_loss = (logsumexp_q - current_q).mean()
    print(f"  ✓ CQL loss: {cql_loss.item():.4f}")
    
    # Check for reasonable values
    if torch.isnan(cql_loss):
        print("  ✗ CQL loss is NaN!")
        return False
    
    if cql_loss.item() > 1000:
        print(f"  ⚠ CQL loss seems very high: {cql_loss.item()}")
        
        # Debug high loss
        print("\n  Debugging high CQL loss:")
        print(f"    Logsumexp Q mean: {logsumexp_q.mean().item():.4f}")
        print(f"    Current Q mean: {current_q.mean().item():.4f}")
        print(f"    All Q-values range: [{all_q.min().item():.4f}, {all_q.max().item():.4f}]")
        print(f"    Masked Q-values range: [{masked_q[valid_masks].min().item():.4f}, {masked_q[valid_masks].max().item():.4f}]")
    
    return True


def test_temperature_scaling():
    """Test different temperature values for logsumexp"""
    print("\nTesting temperature scaling for logsumexp...")
    
    # Create sample Q-values
    q_values = torch.tensor([
        [1.0, 2.0, 3.0, -1e8, -1e8],  # 3 valid actions
        [0.5, 1.5, -1e8, -1e8, -1e8],  # 2 valid actions
    ])
    
    temperatures = [1.0, 5.0, 10.0, 20.0]
    
    for temp in temperatures:
        logsumexp = torch.logsumexp(q_values / temp, dim=1) * temp
        print(f"  Temperature {temp:4.1f}: logsumexp = {logsumexp.tolist()}")
    
    print("\n  Lower temperature makes logsumexp closer to max Q-value")
    print("  Higher temperature makes logsumexp average over all valid Q-values")
    
    return True


def test_batch_update():
    """Test full batch update"""
    print("\nTesting full batch update...")
    
    # Setup
    n_base_features = 17
    action_space = StepwiseActionSpace(max_step=0.1)
    state_dim = n_base_features + action_space.n_vp2_bins
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = UnifiedStepwiseCQL(
        state_dim=state_dim,
        max_step=0.1,
        alpha=0.001,
        gamma=0.95,
        tau=0.8,
        lr=1e-3,
        device=device
    )
    
    batch_size = 32
    
    # Create realistic batch on correct device
    states = torch.randn(batch_size, state_dim, device=device) * 0.1  # Small values
    actions = torch.randint(0, action_space.n_actions, (batch_size,), device=device)
    rewards = torch.randn(batch_size, device=device) * 0.1
    next_states = torch.randn(batch_size, state_dim, device=device) * 0.1
    dones = (torch.rand(batch_size, device=device) < 0.1).float()  # 10% terminal
    current_vp2_doses = torch.rand(batch_size, device=device) * 0.4  # VP2 in [0, 0.4]
    next_vp2_doses = torch.rand(batch_size, device=device) * 0.4
    
    # Run update
    metrics = agent.update(states, actions, rewards, next_states, dones, 
                          current_vp2_doses, next_vp2_doses)
    
    print(f"  Q1 loss: {metrics['q1_loss']:.4f}")
    print(f"  Q2 loss: {metrics['q2_loss']:.4f}")
    print(f"  CQL1 loss: {metrics['cql1_loss']:.4f}")
    print(f"  CQL2 loss: {metrics['cql2_loss']:.4f}")
    
    # Check for reasonable values
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"  ✗ {key} is NaN!")
            return False
        if value > 1000:
            print(f"  ⚠ {key} seems very high: {value}")
    
    print("  ✓ All losses computed successfully")
    return True


def main():
    print("="*60)
    print(" CQL LOSS COMPUTATION UNIT TESTS")
    print("="*60)
    
    tests = [
        test_qnetwork_forward,
        test_cql_loss_computation,
        test_temperature_scaling,
        test_batch_update
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"  ✗ {test.__name__} failed")
        except Exception as e:
            print(f"  ✗ {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f" RESULTS: {passed}/{len(tests)} tests passed")
    print("="*60)
    
    if passed == len(tests):
        print("\n✓ All tests passed! CQL loss computation appears correct.")
        print("\nPotential issues to check:")
        print("  1. Temperature scaling (currently 10.0) might be too high")
        print("  2. Initial Q-network values might be too large")
        print("  3. Learning rate might need adjustment")
    else:
        print("\n✗ Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    main()