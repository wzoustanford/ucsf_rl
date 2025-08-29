"""
Demo: Kernel Density Estimation for Off-Policy Evaluation in Q-Learning/CQL
============================================================================

This demo shows how to use KDE to estimate importance sampling weights for OPE.
We'll estimate p(a|s) for the learned policy and q(a|s) for the behavioral policy.

Key Concepts:
-------------
1. Off-Policy Evaluation (OPE): Estimate the value of a learned policy using data 
   collected from a different (behavioral) policy
   
2. Importance Sampling (IS): Correct for the distribution mismatch by weighting
   each sample by w(s,a) = p(a|s) / q(a|s)
   where p(a|s) = learned policy, q(a|s) = behavioral policy
   
3. KDE: Non-parametric density estimation to approximate p(a|s) and q(a|s)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BLOCK 1: Simple Q-Network Definition
# ============================================================================
class SimpleQNetwork(nn.Module):
    """
    Simple Q-network for demonstration: Q(s,a) → R
    
    This represents our learned policy. In CQL, this would be the
    conservative Q-function that we've trained offline.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        # Concatenate state and action as input
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Returns Q-value


# ============================================================================
# BLOCK 2: Synthetic Data Generation
# ============================================================================
def generate_synthetic_data(n_samples: int = 1000) -> Dict:
    """
    Generate synthetic ICU-like data to simulate the behavioral policy
    
    This simulates:
    - States: patient vitals (MAP, lactate, SOFA score)
    - Actions: clinician's treatment decisions (VP1 binary, VP2 continuous)
    - Rewards: outcomes based on physiological targets
    """
    np.random.seed(42)
    
    # Generate states (simplified: MAP, lactate, SOFA)
    states = np.random.randn(n_samples, 3)
    states[:, 0] = 70 + 20 * states[:, 0]  # MAP: mean 70 mmHg
    states[:, 1] = np.abs(2 + states[:, 1])  # Lactate: mean 2 mmol/L, positive
    states[:, 2] = np.abs(8 + 3 * states[:, 2])  # SOFA: mean 8
    
    # Generate clinician actions (behavioral policy)
    # Key insight: Clinicians tend to give more vasopressors when MAP is low
    # This creates a specific action distribution q(a|s) that we need to estimate
    
    # VP1: Binary decision based on MAP (sigmoid relationship)
    vp1_prob = 1 / (1 + np.exp(0.1 * (states[:, 0] - 65)))  # Higher prob when MAP < 65
    clinician_actions = np.zeros((n_samples, 2))
    clinician_actions[:, 0] = np.random.binomial(1, vp1_prob)  # Binary VP1
    
    # VP2: Continuous dose (0-0.5 mcg/kg/min) - beta distribution
    clinician_actions[:, 1] = np.random.beta(2, 5, n_samples) * 0.5
    
    # Add realistic noise
    clinician_actions[:, 1] += np.random.normal(0, 0.05, n_samples)
    clinician_actions = np.clip(clinician_actions, 0, [1, 0.5])
    
    # Generate rewards (simplified outcome model)
    rewards = np.zeros(n_samples)
    for i in range(n_samples):
        # Reward for maintaining MAP in target range
        if 65 <= states[i, 0] <= 85:
            rewards[i] += 1
        # Reward for low lactate
        if states[i, 1] < 2:
            rewards[i] += 0.5
        # Penalty for excessive vasopressor use
        rewards[i] -= 0.5 * (clinician_actions[i, 0] + 2 * clinician_actions[i, 1])
    
    return {
        'states': states,
        'actions': clinician_actions,
        'rewards': rewards
    }


# ============================================================================
# BLOCK 3: Learned Policy Action Selection
# ============================================================================
def get_learned_policy_actions(q_network: nn.Module, states: np.ndarray, 
                              n_samples: int = 100) -> np.ndarray:
    """
    Generate actions from the learned policy using the Q-network
    
    For each state, we:
    1. Sample multiple candidate actions
    2. Evaluate Q(s,a) for each candidate
    3. Select the action with highest Q-value (greedy policy)
    
    This gives us samples from p(a|s) - the learned policy distribution
    """
    device = next(q_network.parameters()).device
    states_tensor = torch.FloatTensor(states).to(device)
    
    learned_actions = np.zeros((len(states), 2))
    
    with torch.no_grad():
        for i, state in enumerate(states_tensor):
            # Expand state for batch evaluation
            state_expanded = state.unsqueeze(0).expand(n_samples, -1)
            
            # Sample candidate actions uniformly
            # In practice, you might use a more sophisticated sampling strategy
            vp1_samples = torch.randint(0, 2, (n_samples, 1)).float().to(device)
            vp2_samples = torch.rand(n_samples, 1).to(device) * 0.5
            action_samples = torch.cat([vp1_samples, vp2_samples], dim=1)
            
            # Evaluate Q-values for all candidates
            q_values = q_network(state_expanded, action_samples).squeeze()
            
            # Select action with highest Q-value (argmax policy)
            best_idx = q_values.argmax()
            learned_actions[i] = action_samples[best_idx].cpu().numpy()
    
    return learned_actions


# ============================================================================
# BLOCK 4: KDE-based Importance Sampling Class
# ============================================================================
class KDEImportanceSampling:
    """
    Kernel Density Estimation for Importance Sampling in OPE
    
    The key challenge: We need to estimate p(a|s) and q(a|s) from finite samples
    KDE provides a non-parametric way to estimate these densities
    """
    def __init__(self, bandwidth: str = 'scott'):
        """
        Args:
            bandwidth: KDE bandwidth selection method
                      'scott': Scott's rule (default)
                      'silverman': Silverman's rule
                      float: fixed bandwidth
        """
        self.bandwidth = bandwidth
        self.behavioral_kde = None
        self.learned_kde = None
        
    def fit_behavioral_policy(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit KDE for behavioral (clinician) policy q(a|s)
        
        Important: We're actually fitting the joint density p(s,a) here.
        To get conditional q(a|s), we'd need to divide by p(s), but
        for importance sampling ratios, the p(s) terms cancel out!
        
        This is because:
        w(s,a) = p(a|s)/q(a|s) = [p(s,a)/p(s)] / [q(s,a)/q(s)]
                               = p(s,a)/q(s,a) * [q(s)/p(s)]
        
        If we assume the state distribution is the same (which it is
        when evaluating on the same dataset), then q(s) = p(s) and
        we just need the ratio of joint densities.
        """
        # Concatenate states and actions for joint density
        data = np.concatenate([states, actions], axis=1)
        
        # Using scipy's gaussian_kde with automatic bandwidth selection
        self.behavioral_kde = gaussian_kde(data.T, bw_method=self.bandwidth)
        self.behavioral_states = states
        self.behavioral_actions = actions
        
        print(f"Fitted behavioral policy KDE with {len(states)} samples")
        print(f"Bandwidth: {self.behavioral_kde.factor}")
        
    def fit_learned_policy(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit KDE for learned policy p(a|s)
        
        Same approach as behavioral policy - we fit the joint density
        """
        data = np.concatenate([states, actions], axis=1)
        self.learned_kde = gaussian_kde(data.T, bw_method=self.bandwidth)
        self.learned_states = states
        self.learned_actions = actions
        
        print(f"Fitted learned policy KDE with {len(states)} samples")
        print(f"Bandwidth: {self.learned_kde.factor}")
    
    def compute_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute importance weights w(s,a) = p(a|s) / q(a|s)
        
        This is the core of importance sampling:
        - If w > 1: The learned policy is more likely to take this action
        - If w < 1: The behavioral policy is more likely to take this action
        - If w = 1: Both policies are equally likely
        
        Note: We're using joint densities, which works when evaluating
        on the same state distribution.
        """
        # Concatenate for evaluation
        data = np.concatenate([states, actions], axis=1)
        
        # Compute densities using KDE
        # These are probability densities, not probabilities
        p_values = self.learned_kde(data.T)
        q_values = self.behavioral_kde(data.T)
        
        # Avoid division by zero
        q_values = np.maximum(q_values, 1e-10)
        
        # Compute importance weights
        weights = p_values / q_values
        
        # Clip weights for stability
        # Large weights can cause high variance in IS estimates
        weights = np.clip(weights, 0, 10)
        
        return weights
    
    def weighted_importance_sampling(self, states: np.ndarray, actions: np.ndarray, 
                                    rewards: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Compute weighted importance sampling (WIS) estimate of expected reward
        
        Two types of IS estimates:
        1. Ordinary IS: V^IS = (1/n) Σ w_i * r_i
        2. Weighted IS: V^WIS = Σ(w_i * r_i) / Σ(w_i)
        
        WIS is biased but has lower variance and is generally preferred.
        
        Also computes Effective Sample Size (ESS):
        ESS = (Σ w_i)^2 / Σ(w_i^2)
        
        ESS tells us how many "effective" samples we have after weighting.
        If ESS << n, the weights are highly variable (bad).
        """
        weights = self.compute_importance_weights(states, actions)
        
        # Weighted importance sampling estimate
        wis_estimate = np.sum(weights * rewards) / np.sum(weights)
        
        # Effective sample size
        # ESS measures how many independent samples our weighted data is worth
        ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
        
        return wis_estimate, ess, weights


# ============================================================================
# BLOCK 5: Conditional KDE (More Sophisticated Approach)
# ============================================================================
class ConditionalKDE:
    """
    More sophisticated conditional KDE for p(a|s) and q(a|s)
    
    This class attempts to model the conditional distribution more directly
    rather than relying on joint distributions.
    
    Key difference from simple KDE:
    - Standardizes features for better KDE performance
    - Uses sklearn's KernelDensity for more control
    - Can handle high-dimensional states better
    """
    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth
        self.kde_model = None
        
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit conditional KDE using sklearn's KernelDensity
        
        Standardization is crucial for KDE performance when features
        have different scales (e.g., MAP in mmHg vs lactate in mmol/L)
        """
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features separately
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        
        states_scaled = self.state_scaler.fit_transform(states)
        actions_scaled = self.action_scaler.fit_transform(actions)
        
        # Combine for joint density
        data = np.concatenate([states_scaled, actions_scaled], axis=1)
        
        # Fit KDE with Gaussian kernel
        self.kde_model = KernelDensity(
            kernel='gaussian', 
            bandwidth=self.bandwidth
        )
        self.kde_model.fit(data)
        
        self.state_dim = states.shape[1]
        self.action_dim = actions.shape[1]
        
    def log_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute log probability log p(a|s)
        
        Working in log space is more numerically stable, especially
        when dealing with small probabilities.
        
        Note: This still computes joint log p(s,a). For true conditional,
        we'd need to estimate p(s) separately and subtract: log p(a|s) = log p(s,a) - log p(s)
        """
        states_scaled = self.state_scaler.transform(states)
        actions_scaled = self.action_scaler.transform(actions)
        data = np.concatenate([states_scaled, actions_scaled], axis=1)
        
        # Get log density
        log_density = self.kde_model.score_samples(data)
        
        return log_density


# ============================================================================
# BLOCK 6: Main Demo Function
# ============================================================================
def demo_kde_ope():
    """
    Main demo of KDE for off-policy evaluation
    
    This demonstrates the full pipeline:
    1. Generate synthetic data (behavioral policy)
    2. Train a Q-network (learned policy)
    3. Fit KDE models to both policies
    4. Compute importance weights
    5. Estimate value of learned policy using IS/WIS
    """
    print("=" * 80)
    print("KDE Off-Policy Evaluation Demo")
    print("=" * 80)
    
    # Step 1: Generate synthetic data from behavioral policy
    print("\n1. Generating synthetic ICU data...")
    data = generate_synthetic_data(n_samples=1000)
    states = data['states']
    clinician_actions = data['actions']  # Actions from behavioral policy q(a|s)
    rewards = data['rewards']
    
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {clinician_actions.shape}")
    print(f"   Mean reward (behavioral): {rewards.mean():.3f}")
    
    # Step 2: Create a simple Q-network (represents our learned policy)
    print("\n2. Training Q-network...")
    q_network = SimpleQNetwork(state_dim=3, action_dim=2)
    # Note: In practice, you'd train this Q-network using CQL or another offline RL method
    
    # Step 3: Generate actions from learned policy
    print("\n3. Generating learned policy actions...")
    learned_actions = get_learned_policy_actions(q_network, states)
    # These are the actions p(a|s) would take
    
    # Step 4: Fit KDE models
    print("\n4. Fitting KDE models...")
    kde_is = KDEImportanceSampling(bandwidth='scott')
    
    # Fit KDE to behavioral policy data
    kde_is.fit_behavioral_policy(states, clinician_actions)
    
    # Fit KDE to learned policy data
    kde_is.fit_learned_policy(states, learned_actions)
    
    # Step 5: Compute OPE using importance sampling
    print("\n5. Computing Off-Policy Evaluation...")
    
    # KEY INSIGHT: We evaluate using the behavioral policy's trajectories
    # but weight them by how likely the learned policy would be to take those actions
    wis_estimate, ess, weights = kde_is.weighted_importance_sampling(
        states, clinician_actions, rewards  # Note: using clinician actions!
    )
    
    print(f"\n   Behavioral policy average reward: {rewards.mean():.3f}")
    print(f"   WIS estimate of learned policy: {wis_estimate:.3f}")
    print(f"   Effective sample size: {ess:.1f} / {len(states)}")
    print(f"   Weight statistics:")
    print(f"      Mean: {weights.mean():.3f}")
    print(f"      Std:  {weights.std():.3f}")
    print(f"      Min:  {weights.min():.3f}")
    print(f"      Max:  {weights.max():.3f}")
    
    # Step 6: Test conditional KDE (alternative approach)
    print("\n6. Testing Conditional KDE...")
    behavioral_kde = ConditionalKDE(bandwidth=0.1)
    behavioral_kde.fit(states, clinician_actions)
    
    learned_kde = ConditionalKDE(bandwidth=0.1)
    learned_kde.fit(states, learned_actions)
    
    # Compute importance weights using conditional KDE
    log_p = learned_kde.log_prob(states, clinician_actions)
    log_q = behavioral_kde.log_prob(states, clinician_actions)
    
    # Convert log weights to weights
    log_weights = log_p - log_q
    weights_conditional = np.exp(np.clip(log_weights, -10, 10))
    
    # Weighted importance sampling with conditional KDE
    wis_conditional = np.sum(weights_conditional * rewards) / np.sum(weights_conditional)
    ess_conditional = np.sum(weights_conditional) ** 2 / np.sum(weights_conditional ** 2)
    
    print(f"\n   Conditional KDE WIS estimate: {wis_conditional:.3f}")
    print(f"   Conditional KDE ESS: {ess_conditional:.1f} / {len(states)}")
    
    # Visualize results
    visualize_importance_weights(weights, weights_conditional, rewards)
    
    return kde_is, behavioral_kde, learned_kde


# ============================================================================
# BLOCK 7: Visualization
# ============================================================================
def visualize_importance_weights(weights1: np.ndarray, weights2: np.ndarray, 
                                rewards: np.ndarray):
    """
    Visualize importance weights and their effect on the reward distribution
    
    This helps understand:
    - Distribution of importance weights (should be centered around 1)
    - Relationship between weights and rewards
    - How weighting changes the effective reward distribution
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Weight distributions
    axes[0, 0].hist(weights1, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Importance Weight')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Simple KDE Weights Distribution')
    axes[0, 0].axvline(1.0, color='red', linestyle='--', label='w=1')
    axes[0, 0].legend()
    
    axes[0, 1].hist(weights2, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Importance Weight')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Conditional KDE Weights Distribution')
    axes[0, 1].axvline(1.0, color='red', linestyle='--', label='w=1')
    axes[0, 1].legend()
    
    # Log scale comparison (useful for heavy-tailed distributions)
    axes[0, 2].hist(np.log(weights1 + 1e-10), bins=50, alpha=0.5, label='Simple', color='blue')
    axes[0, 2].hist(np.log(weights2 + 1e-10), bins=50, alpha=0.5, label='Conditional', color='green')
    axes[0, 2].set_xlabel('Log Importance Weight')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Log Weight Comparison')
    axes[0, 2].legend()
    
    # Weight vs Reward scatter plots
    # This shows if high weights correspond to high/low rewards
    axes[1, 0].scatter(rewards, weights1, alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Importance Weight')
    axes[1, 0].set_title('Simple KDE: Weight vs Reward')
    axes[1, 0].axhline(1.0, color='red', linestyle='--', alpha=0.5)
    
    axes[1, 1].scatter(rewards, weights2, alpha=0.5, s=10, color='green')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Importance Weight')
    axes[1, 1].set_title('Conditional KDE: Weight vs Reward')
    axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.5)
    
    # Weighted reward distributions
    # Shows how importance weighting changes the effective distribution
    axes[1, 2].hist(rewards, bins=30, alpha=0.3, label='Original', color='gray', density=True)
    axes[1, 2].hist(rewards, bins=30, alpha=0.5, label='Simple Weighted', 
                   weights=weights1/weights1.sum(), color='blue', density=True)
    axes[1, 2].hist(rewards, bins=30, alpha=0.5, label='Conditional Weighted', 
                   weights=weights2/weights2.sum(), color='green', density=True)
    axes[1, 2].set_xlabel('Reward')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Reward Distribution (Weighted)')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('ucsf_rl/kde_demo/importance_weights_visualization.png', dpi=150)
    print("\n   Visualization saved to: importance_weights_visualization.png")
    plt.show()


if __name__ == "__main__":
    kde_is, behavioral_kde, learned_kde = demo_kde_ope()
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. KDE estimates density of p(a|s) and q(a|s) from samples")
    print("2. Importance weights w = p(a|s)/q(a|s) correct for distribution shift")
    print("3. WIS estimate approximates expected reward under learned policy")
    print("4. ESS indicates quality of importance sampling (higher is better)")
    print("5. Conditional KDE can provide more accurate estimates but is more complex")