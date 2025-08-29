"""
LSTM Q-Network for Sequential CQL
==================================
Implements LSTM-based Q-networks compatible with the existing CQL architecture
but capable of processing sequences from the medical_sequence_buffer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class LSTMQNetwork(nn.Module):
    """
    Q-Network with LSTM for sequential medical decision making.
    Compatible with both binary and continuous action spaces.
    
    Architecture:
    - Feature extraction layers (same as existing CQL)
    - LSTM for temporal modeling
    - Output layer for Q-values
    
    For binary actions: outputs Q(s,a) where a is concatenated to state
    For compatibility with existing CQL training.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,  # 1 for binary, 2 for dual
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM Q-Network.
        
        Args:
            state_dim: Dimension of state space (18 for medical data)
            action_dim: Dimension of action space (1 for binary)
            hidden_dim: Hidden dimension for feature extraction
            lstm_hidden: LSTM hidden state dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        
        # Feature extraction layers (similar to existing CQL networks)
        # Input: state + action concatenated
        input_dim = state_dim + action_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer - single Q-value for the state-action pair
        self.q_head = nn.Linear(lstm_hidden, 1)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM Q-network.
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            actions: [batch_size, sequence_length, action_dim]
            hidden_state: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: [batch_size, sequence_length, 1] - Q(s,a) values
            new_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len, _ = states.shape
        
        # Concatenate states and actions
        state_action = torch.cat([states, actions], dim=-1)
        
        # Reshape for feature extraction
        state_action_flat = state_action.reshape(-1, self.state_dim + self.action_dim)
        features_flat = self.feature_extractor(state_action_flat)
        features = features_flat.reshape(batch_size, seq_len, -1)
        
        # Process through LSTM
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, states.device)
        
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Compute Q-values
        q_values = self.q_head(lstm_out)  # [batch_size, seq_len, 1]
        
        return q_values, new_hidden
    
    def forward_single_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single timestep (useful for inference).
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            hidden_state: Optional LSTM hidden state
        
        Returns:
            q_value: [batch_size, 1] - Q(s,a) value
            new_hidden: Updated LSTM hidden state
        """
        # Add sequence dimension
        state_seq = state.unsqueeze(1)  # [batch_size, 1, state_dim]
        action_seq = action.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Forward through network
        q_values_seq, new_hidden = self.forward(state_seq, action_seq, hidden_state)
        
        # Remove sequence dimension
        q_value = q_values_seq.squeeze(1)  # [batch_size, 1]
        
        return q_value, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden).to(device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden).to(device)
        return (h0, c0)
    
    def get_action_values(
        self,
        states: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get Q-values for both actions (0 and 1) for binary action space.
        Used for action selection.
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            hidden_state: Optional LSTM hidden state
        
        Returns:
            q_values: [batch_size, sequence_length, 2] - Q-values for actions 0 and 1
            new_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len, _ = states.shape
        
        # Create action tensors for both possible actions
        actions_0 = torch.zeros(batch_size, seq_len, self.action_dim).to(states.device)
        actions_1 = torch.ones(batch_size, seq_len, self.action_dim).to(states.device)
        
        # Get Q-values for both actions
        q_values_0, hidden_0 = self.forward(states, actions_0, hidden_state)
        q_values_1, hidden_1 = self.forward(states, actions_1, hidden_state)
        
        # Combine Q-values
        q_values = torch.cat([q_values_0, q_values_1], dim=-1)  # [batch_size, seq_len, 2]
        
        # Use the hidden state from action 0 (arbitrary choice, both should be similar)
        return q_values, hidden_0


class LSTMBinaryCQL:
    """
    Binary CQL with LSTM Q-networks for sequential learning.
    Adapts the existing BinaryCQL to work with sequences.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.005,
        lr: float = 3e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize LSTM Binary CQL."""
        self.state_dim = state_dim
        self.action_dim = 1  # Binary action
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        
        # Create LSTM Q-networks
        self.q1 = LSTMQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        self.q2 = LSTMQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        # Target networks
        self.q1_target = LSTMQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        self.q2_target = LSTMQNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
    
    def compute_cql_loss_sequences(
        self,
        q_network: LSTMQNetwork,
        q_target_1: LSTMQNetwork,
        q_target_2: LSTMQNetwork,
        burn_in_states: torch.Tensor,
        burn_in_actions: torch.Tensor,
        training_states: torch.Tensor,
        training_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CQL loss for sequences.
        
        Args:
            q_network: Q-network to update
            q_target_1, q_target_2: Target networks
            burn_in_states: [batch_size, burn_in_len, state_dim]
            burn_in_actions: [batch_size, burn_in_len, action_dim]
            training_states: [batch_size, training_len, state_dim]
            training_actions: [batch_size, training_len, action_dim]
            rewards: [batch_size, training_len]
            next_states: [batch_size, training_len, state_dim]
            dones: [batch_size, training_len]
            weights: [batch_size] - importance sampling weights
        """
        batch_size = burn_in_states.shape[0]
        
        # Burn-in phase: warm up LSTM hidden state (no gradients)
        with torch.no_grad():
            hidden = q_network.init_hidden(batch_size, self.device)
            _, hidden = q_network(burn_in_states, burn_in_actions, hidden)
        
        # Forward pass through training sequence
        q_values, _ = q_network(training_states, training_actions, hidden)
        q_values = q_values.squeeze(-1)  # [batch_size, training_len]
        
        # Compute target Q-values
        with torch.no_grad():
            # Get Q-values for both actions at next states
            next_q1_0, _ = q_target_1(next_states, torch.zeros_like(training_actions), hidden)
            next_q1_1, _ = q_target_1(next_states, torch.ones_like(training_actions), hidden)
            next_q2_0, _ = q_target_2(next_states, torch.zeros_like(training_actions), hidden)
            next_q2_1, _ = q_target_2(next_states, torch.ones_like(training_actions), hidden)
            
            # Take minimum of two Q-networks and maximum over actions
            next_q1 = torch.max(torch.cat([next_q1_0, next_q1_1], dim=-1), dim=-1)[0]
            next_q2 = torch.max(torch.cat([next_q2_0, next_q2_1], dim=-1), dim=-1)[0]
            next_q = torch.min(next_q1, next_q2)
            
            # Compute targets
            targets = rewards + self.gamma * next_q * (1 - dones)
        
        # TD loss with importance sampling weights
        td_loss = (q_values - targets) ** 2
        td_loss = td_loss.mean(dim=1)  # Average over sequence
        td_loss = (td_loss * weights).mean()
        
        # CQL regularization
        # Sample random actions for comparison
        random_actions = torch.randint(0, 2, training_actions.shape).float().to(self.device)
        random_q_values, _ = q_network(training_states, random_actions, hidden)
        random_q_values = random_q_values.squeeze(-1)
        
        # Conservative penalty: penalize Q-values for random actions
        cql_loss = (random_q_values - q_values).mean()
        
        # Total loss
        total_loss = td_loss + self.alpha * cql_loss
        
        metrics = {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'q_values': q_values.mean().item(),
            'targets': targets.mean().item()
        }
        
        return total_loss, metrics
    
    def update_sequences(
        self,
        burn_in_batch: Dict[str, torch.Tensor],
        training_batch: Dict[str, torch.Tensor],
        weights: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks using sequence batch."""
        
        # Update Q1
        q1_loss, q1_metrics = self.compute_cql_loss_sequences(
            self.q1, self.q1_target, self.q2_target,
            burn_in_batch['states'],
            burn_in_batch['actions'],
            training_batch['states'],
            training_batch['actions'],
            training_batch['rewards'],
            training_batch['next_states'],
            training_batch['dones'],
            weights
        )
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()
        
        # Update Q2
        q2_loss, q2_metrics = self.compute_cql_loss_sequences(
            self.q2, self.q1_target, self.q2_target,
            burn_in_batch['states'],
            burn_in_batch['actions'],
            training_batch['states'],
            training_batch['actions'],
            training_batch['rewards'],
            training_batch['next_states'],
            training_batch['dones'],
            weights
        )
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()
        
        # Soft update target networks
        self.soft_update_targets()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_td_loss': q1_metrics['td_loss'],
            'q1_cql_loss': q1_metrics['cql_loss'],
            'q_values': q1_metrics['q_values']
        }
    
    def soft_update_targets(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Test function
if __name__ == "__main__":
    print("Testing LSTM Q-Network...")
    
    # Test parameters
    batch_size = 32
    burn_in_len = 8
    training_len = 12
    state_dim = 18
    action_dim = 1
    
    # Create network
    net = LSTMQNetwork(state_dim=state_dim, action_dim=action_dim)
    
    # Create dummy data
    burn_in_states = torch.randn(batch_size, burn_in_len, state_dim)
    burn_in_actions = torch.randint(0, 2, (batch_size, burn_in_len, action_dim)).float()
    training_states = torch.randn(batch_size, training_len, state_dim)
    training_actions = torch.randint(0, 2, (batch_size, training_len, action_dim)).float()
    
    # Test burn-in
    with torch.no_grad():
        hidden = net.init_hidden(batch_size, burn_in_states.device)
        _, hidden = net(burn_in_states, burn_in_actions, hidden)
        print(f"✓ Burn-in complete. Hidden shape: {hidden[0].shape}")
    
    # Test forward pass
    q_values, new_hidden = net(training_states, training_actions, hidden)
    print(f"✓ Forward pass complete. Q-values shape: {q_values.shape}")
    
    # Test action selection
    q_both_actions, _ = net.get_action_values(training_states, hidden)
    print(f"✓ Action values computed. Shape: {q_both_actions.shape}")
    
    # Test CQL
    print("\nTesting LSTM Binary CQL...")
    cql = LSTMBinaryCQL(state_dim=state_dim, device='cpu')
    
    # Create full batch
    rewards = torch.randn(batch_size, training_len)
    next_states = torch.randn(batch_size, training_len, state_dim)
    dones = torch.zeros(batch_size, training_len)
    weights = torch.ones(batch_size)
    
    # Test update
    burn_in_batch = {'states': burn_in_states, 'actions': burn_in_actions}
    training_batch = {
        'states': training_states,
        'actions': training_actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones
    }
    
    metrics = cql.update_sequences(burn_in_batch, training_batch, weights)
    print(f"✓ CQL update complete. Metrics: {metrics}")
    
    print("\n✓ All tests passed!")