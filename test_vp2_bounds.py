#!/usr/bin/env python3
"""
Test VP2 bounds with minimum value of 0.05
"""

import numpy as np
from run_unified_stepwise_cql_allalphas import StepwiseActionSpace

# Initialize action space
action_space = StepwiseActionSpace(max_step=0.1)

print("="*60)
print(" VP2 BOUNDS TEST")
print("="*60)

print(f"\nVP2 bounds:")
print(f"  VP2_MIN: {action_space.VP2_MIN}")
print(f"  VP2_MAX: {action_space.VP2_MAX}")
print(f"  MIN_STEP: {action_space.MIN_STEP}")

print(f"\nVP2 bins:")
print(f"  Number of bins: {action_space.n_vp2_bins}")
print(f"  Bin centers: {action_space.vp2_bin_centers}")

print(f"\nVP2 changes: {action_space.VP2_CHANGES}")

# Test boundary cases
test_doses = [0.0, 0.025, 0.05, 0.1, 0.45, 0.5, 0.55]
print(f"\nTesting vp2_to_bin():")
for dose in test_doses:
    bin_idx = action_space.vp2_to_bin(dose)
    print(f"  VP2 {dose:.3f} -> bin {bin_idx}")

# Test apply_action at boundaries
print(f"\nTesting apply_action() at boundaries:")
test_cases = [
    (0.05, -0.1, "At minimum, try to decrease"),
    (0.05, 0.0, "At minimum, no change"),
    (0.05, 0.05, "At minimum, try to increase"),
    (0.5, -0.05, "At maximum, try to decrease"),
    (0.5, 0.0, "At maximum, no change"),
    (0.5, 0.05, "At maximum, try to increase"),
]

for current_vp2, change, desc in test_cases:
    # Find action index for VP1=1 and the given change
    change_idx = np.argmin(np.abs(action_space.VP2_CHANGES - change))
    action_idx = action_space.get_discrete_action(1, change_idx)
    vp1, new_vp2 = action_space.apply_action(action_idx, current_vp2)
    print(f"  {desc}:")
    print(f"    Current: {current_vp2:.2f}, Change: {change:+.2f} -> New: {new_vp2:.2f}")

# Test valid actions at boundaries
print(f"\nTesting valid actions at boundaries:")
for vp2 in [0.05, 0.1, 0.45, 0.5]:
    valid = action_space.get_valid_actions(vp2)
    n_valid = valid.sum()
    print(f"  VP2={vp2:.2f}: {n_valid}/{action_space.n_actions} actions valid")
    
    # Show which VP2 changes are valid
    valid_changes = []
    for i in range(action_space.n_actions):
        if valid[i]:
            _, vp2_change_idx = action_space.decode_action(i)
            valid_changes.append(action_space.VP2_CHANGES[vp2_change_idx])
    unique_changes = sorted(set(valid_changes))
    print(f"    Valid VP2 changes: {unique_changes}")

print("\n✓ VP2 bounds test complete!")
print(f"  VP2 is now constrained to [{action_space.VP2_MIN}, {action_space.VP2_MAX}]")
print(f"  This ensures VP2 is always > 0")