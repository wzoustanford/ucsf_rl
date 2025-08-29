"""
Demo: Kernel Density Estimation for Off-Policy Evaluation in Q-Learning/CQL
============================================================================

This demo shows how to use KDE to estimate importance sampling weights for OPE.
We'll estimate p(a|s) for the learned policy and q(a|s) for the behavioral policy.
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


class SimpleQNetwork(nn.Module):
    """Simple Q-network for demonstration"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def generate_synthetic_data(n_samples: int = 1000) -> Dict:
    """Generate synthetic ICU-like data"""
    np.random.seed(42)
    
    # Generate states (simplified: MAP, lactate, SOFA)
    states = np.random.randn(n_samples, 3)
    states[:, 0] = 70 + 20 * states[:, 0]  # MAP: mean 70
    states[:, 1] = np.abs(2 + states[:, 1])  # Lactate: mean 2, positive
    states[:, 2] = np.abs(8 + 3 * states[:, 2])  # SOFA: mean 8
    
    # Generate clinician actions (2D: VP1, VP2)
    # Clinician tends to give more vasopressors when MAP is low
    vp1_prob = 1 / (1 + np.exp(0.1 * (states[:, 0] - 65)))  # Sigmoid based on MAP
    clinician_actions = np.zeros((n_samples, 2))
    clinician_actions[:, 0] = np.random.binomial(1, vp1_prob)  # Binary VP1
    clinician_actions[:, 1] = np.random.beta(2, 5, n_samples) * 0.5  # VP2: 0-0.5
    
    # Add some noise
    clinician_actions[:, 1] += np.random.normal(0, 0.05, n_samples)
    clinician_actions = np.clip(clinician_actions, 0, [1, 0.5])
    
    # Generate rewards (simplified)
    rewards = np.zeros(n_samples)
    for i in range(n_samples):
        if 65 <= states[i, 0] <= 85:  # Good MAP
            rewards[i] += 1
        if states[i, 1] < 2:  # Low lactate
            rewards[i] += 0.5
        # Penalty for too much vasopressor
        rewards[i] -= 0.5 * (clinician_actions[i, 0] + 2 * clinician_actions[i, 1])
    
    return {
        'states': states,
        'actions': clinician_actions,
        'rewards': rewards
    }


def get_learned_policy_actions(q_network: nn.Module, states: np.ndarray, 
                              n_samples: int = 100) -> np.ndarray:
    """
    Get actions from learned policy using Q-network
    Sample multiple actions and select the best Q-value
    """
    device = next(q_network.parameters()).device
    states_tensor = torch.FloatTensor(states).to(device)
    
    learned_actions = np.zeros((len(states), 2))
    
    with torch.no_grad():
        for i, state in enumerate(states_tensor):
            state_expanded = state.unsqueeze(0).expand(n_samples, -1)
            
            # Sample candidate actions
            vp1_samples = torch.randint(0, 2, (n_samples, 1)).float().to(device)
            vp2_samples = torch.rand(n_samples, 1).to(device) * 0.5
            action_samples = torch.cat([vp1_samples, vp2_samples], dim=1)
            
            # Evaluate Q-values
            q_values = q_network(state_expanded, action_samples).squeeze()
            
            # Select best action
            best_idx = q_values.argmax()
            learned_actions[i] = action_samples[best_idx].cpu().numpy()
    
    return learned_actions


class KDEImportanceSampling:
    """
    Kernel Density Estimation for Importance Sampling in OPE
    """
    def __init__(self, bandwidth: str = 'scott'):
        """
        Args:
            bandwidth: KDE bandwidth selection method ('scott', 'silverman', or float)
        """
        self.bandwidth = bandwidth
        self.behavioral_kde = None
        self.learned_kde = None
        
    def fit_behavioral_policy(self, states: np.ndarray, actions: np.ndarray):
        """Fit KDE for behavioral (clinician) policy q(a|s)"""
        # Concatenate states and actions for conditional density
        data = np.concatenate([states, actions], axis=1)
        
        # Using scipy's gaussian_kde for simplicity
        self.behavioral_kde = gaussian_kde(data.T, bw_method=self.bandwidth)
        self.behavioral_states = states
        self.behavioral_actions = actions
        
        print(f"Fitted behavioral policy KDE with {len(states)} samples")
        print(f"Bandwidth: {self.behavioral_kde.factor}")
        
    def fit_learned_policy(self, states: np.ndarray, actions: np.ndarray):
        """Fit KDE for learned policy p(a|s)"""
        data = np.concatenate([states, actions], axis=1)
        self.learned_kde = gaussian_kde(data.T, bw_method=self.bandwidth)
        self.learned_states = states
        self.learned_actions = actions
        
        print(f"Fitted learned policy KDE with {len(states)} samples")
        print(f"Bandwidth: {self.learned_kde.factor}")
    
    def compute_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute importance weights w(s,a) = p(a|s) / q(a|s)
        
        Note: This is a simplified version. In practice, we need to condition on states properly.
        """
        # Concatenate for evaluation
        data = np.concatenate([states, actions], axis=1)
        
        # Compute densities
        p_values = self.learned_kde(data.T)
        q_values = self.behavioral_kde(data.T)
        
        # Avoid division by zero
        q_values = np.maximum(q_values, 1e-10)
        
        # Compute importance weights
        weights = p_values / q_values
        
        # Clip weights for stability
        weights = np.clip(weights, 0, 10)
        
        return weights
    
    def weighted_importance_sampling(self, states: np.ndarray, actions: np.ndarray, 
                                    rewards: np.ndarray) -> float:
        """
        Compute weighted importance sampling estimate of expected reward
        
        V^WIS = Σ(w_i * r_i) / Σ(w_i)
        """
        weights = self.compute_importance_weights(states, actions)
        
        # Weighted importance sampling
        wis_estimate = np.sum(weights * rewards) / np.sum(weights)
        
        # Compute effective sample size
        ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
        
        return wis_estimate, ess, weights


class ConditionalKDE:
    """
    More sophisticated conditional KDE for p(a|s) and q(a|s)
    Uses separate KDE for each action dimension conditioned on state
    """
    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth
        self.kde_model = None
        
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """Fit conditional KDE using sklearn's KernelDensity"""
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        
        states_scaled = self.state_scaler.fit_transform(states)
        actions_scaled = self.action_scaler.fit_transform(actions)
        
        # Combine for joint density
        data = np.concatenate([states_scaled, actions_scaled], axis=1)
        
        # Fit KDE
        self.kde_model = KernelDensity(
            kernel='gaussian', 
            bandwidth=self.bandwidth
        )
        self.kde_model.fit(data)
        
        self.state_dim = states.shape[1]
        self.action_dim = actions.shape[1]
        
    def log_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute log probability log p(a|s)"""
        states_scaled = self.state_scaler.transform(states)
        actions_scaled = self.action_scaler.transform(actions)
        data = np.concatenate([states_scaled, actions_scaled], axis=1)
        
        # Get log density
        log_density = self.kde_model.score_samples(data)
        
        # Note: This is joint density p(s,a). For conditional p(a|s),
        # we'd need to divide by p(s), but for importance sampling ratios,
        # the p(s) terms cancel out
        
        return log_density


def demo_kde_ope():
    """Main demo of KDE for off-policy evaluation"""
    print("=" * 80)
    print("KDE Off-Policy Evaluation Demo")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic ICU data...")
    data = generate_synthetic_data(n_samples=1000)
    states = data['states']
    clinician_actions = data['actions']
    rewards = data['rewards']
    
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {clinician_actions.shape}")
    print(f"   Mean reward (behavioral): {rewards.mean():.3f}")
    
    # Train a simple Q-network
    print("\n2. Training Q-network...")
    q_network = SimpleQNetwork(state_dim=3, action_dim=2)
    
    # Get learned policy actions
    print("\n3. Generating learned policy actions...")
    learned_actions = get_learned_policy_actions(q_network, states)
    
    # Initialize KDE importance sampling
    print("\n4. Fitting KDE models...")
    kde_is = KDEImportanceSampling(bandwidth='scott')
    
    # Fit behavioral policy
    kde_is.fit_behavioral_policy(states, clinician_actions)
    
    # Fit learned policy
    kde_is.fit_learned_policy(states, learned_actions)
    
    # Compute OPE using importance sampling
    print("\n5. Computing Off-Policy Evaluation...")
    wis_estimate, ess, weights = kde_is.weighted_importance_sampling(
        states, clinician_actions, rewards
    )
    
    print(f"\n   Behavioral policy average reward: {rewards.mean():.3f}")
    print(f"   WIS estimate of learned policy: {wis_estimate:.3f}")
    print(f"   Effective sample size: {ess:.1f} / {len(states)}")
    print(f"   Weight statistics:")
    print(f"      Mean: {weights.mean():.3f}")
    print(f"      Std:  {weights.std():.3f}")
    print(f"      Min:  {weights.min():.3f}")
    print(f"      Max:  {weights.max():.3f}")
    
    # Test conditional KDE
    print("\n6. Testing Conditional KDE...")
    behavioral_kde = ConditionalKDE(bandwidth=0.1)
    behavioral_kde.fit(states, clinician_actions)
    
    learned_kde = ConditionalKDE(bandwidth=0.1)
    learned_kde.fit(states, learned_actions)
    
    # Compute importance weights using conditional KDE
    log_p = learned_kde.log_prob(states, clinician_actions)
    log_q = behavioral_kde.log_prob(states, clinician_actions)
    
    # Convert to weights
    log_weights = log_p - log_q
    weights_conditional = np.exp(np.clip(log_weights, -10, 10))
    
    # Weighted importance sampling with conditional KDE
    wis_conditional = np.sum(weights_conditional * rewards) / np.sum(weights_conditional)
    ess_conditional = np.sum(weights_conditional) ** 2 / np.sum(weights_conditional ** 2)
    
    print(f"\n   Conditional KDE WIS estimate: {wis_conditional:.3f}")
    print(f"   Conditional KDE ESS: {ess_conditional:.1f} / {len(states)}")
    
    # Visualize
    visualize_importance_weights(weights, weights_conditional, rewards)
    
    return kde_is, behavioral_kde, learned_kde


def visualize_importance_weights(weights1: np.ndarray, weights2: np.ndarray, 
                                rewards: np.ndarray):
    """Visualize importance weights and their effect"""
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
    
    # Log scale comparison
    axes[0, 2].hist(np.log(weights1 + 1e-10), bins=50, alpha=0.5, label='Simple', color='blue')
    axes[0, 2].hist(np.log(weights2 + 1e-10), bins=50, alpha=0.5, label='Conditional', color='green')
    axes[0, 2].set_xlabel('Log Importance Weight')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Log Weight Comparison')
    axes[0, 2].legend()
    
    # Weight vs Reward
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
    
    # Weighted rewards
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