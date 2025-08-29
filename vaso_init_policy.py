"""
Vasopressor Initiation Policies
================================
Wrappers that enforce VP1 persistence constraint during evaluation.
Once VP1 is initiated for a patient, it cannot be stopped.
This reflects clinical practice where vasopressor discontinuation is rare.
"""

import numpy as np
import torch


class VasopressorInitiationPolicy:
    """
    Wrapper for BinaryCQL that enforces VP1 persistence constraint.
    Once VP1 is initiated for a patient, all subsequent actions are VP1=1.
    """
    
    def __init__(self, binary_cql_agent):
        """
        Args:
            binary_cql_agent: Trained BinaryCQL agent
        """
        self.agent = binary_cql_agent
        self.device = self.agent.device
        self.agent.q1.eval()
        self.agent.q2.eval()
        
        # Track VP1 initiation status for each patient
        self.vp1_initiated = {}
        
    def reset(self):
        """Reset VP1 tracking for new evaluation"""
        self.vp1_initiated = {}
    
    def select_action(self, state: np.ndarray, patient_id: int, 
                     epsilon: float = 0.0) -> int:
        """
        Select action with VP1 persistence constraint
        
        Args:
            state: Current state
            patient_id: Patient identifier
            epsilon: Epsilon for epsilon-greedy (usually 0 for evaluation)
            
        Returns:
            Action (0 or 1)
        """
        # Check if VP1 already initiated for this patient
        if patient_id in self.vp1_initiated and self.vp1_initiated[patient_id]:
            return 1  # Must continue VP1
        
        # Use the agent's select_action method
        action_array = self.agent.select_action(state, epsilon=epsilon)
        action = int(action_array[0])
        
        # If VP1 is initiated, mark it for this patient
        if action == 1:
            self.vp1_initiated[patient_id] = True
        
        return action
    
    def get_q_values(self, state: np.ndarray) -> tuple:
        """
        Get Q-values for both actions
        
        Args:
            state: Current state
            
        Returns:
            (q0, q1) - Q-values for action 0 and 1
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get Q-values for both actions
            q0_1 = self.agent.q1(state_t, torch.FloatTensor([[0]]).to(self.device)).item()
            q1_1 = self.agent.q1(state_t, torch.FloatTensor([[1]]).to(self.device)).item()
            
            q0_2 = self.agent.q2(state_t, torch.FloatTensor([[0]]).to(self.device)).item()
            q1_2 = self.agent.q2(state_t, torch.FloatTensor([[1]]).to(self.device)).item()
            
            # Use minimum for conservative estimate
            q0 = min(q0_1, q0_2)
            q1 = min(q1_1, q1_2)
            
            return q0, q1


class DualVasopressorInitiationPolicy:
    """
    Wrapper for DualMixedCQL that enforces VP1 persistence constraint.
    Once VP1 is initiated for a patient, VP1 stays on (=1).
    VP2 continues to be controlled by the model.
    """
    
    def __init__(self, dual_agent):
        """
        Args:
            dual_agent: Trained DualMixedCQL agent
        """
        self.agent = dual_agent
        self.device = self.agent.device
        self.agent.q1.eval()
        self.agent.q2.eval()
        
        # Track VP1 initiation status for each patient
        self.vp1_initiated = {}
        
    def reset(self):
        """Reset VP1 tracking for new evaluation"""
        self.vp1_initiated = {}
    
    def select_action(self, state: np.ndarray, patient_id: int) -> np.ndarray:
        """
        Select action with VP1 persistence constraint
        
        Args:
            state: Current state
            patient_id: Patient identifier
            
        Returns:
            [VP1, VP2] where VP1 is binary, VP2 is continuous [0, 0.5]
        """
        # Get model's suggested action
        model_action = self.agent.select_action(state)
        
        # Check if VP1 already initiated for this patient
        if patient_id in self.vp1_initiated and self.vp1_initiated[patient_id]:
            model_action[0] = 1  # Force VP1 to continue
        elif model_action[0] > 0:
            # If VP1 is initiated, mark it for this patient
            self.vp1_initiated[patient_id] = True
            
        return model_action
    
    def get_q_value(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Get Q-value for a state-action pair
        
        Args:
            state: Current state
            action: [VP1, VP2] action
            
        Returns:
            Q-value (minimum of Q1 and Q2)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Check if agent is Block Discrete (has continuous_to_discrete_action method)
            if hasattr(self.agent, 'continuous_to_discrete_action'):
                # Block Discrete model - convert continuous action to discrete index
                action_idx = self.agent.continuous_to_discrete_action(action)
                action_idx_t = torch.LongTensor([action_idx]).to(self.device)
                q1 = self.agent.q1(state_t, action_idx_t).item()
                q2 = self.agent.q2(state_t, action_idx_t).item()
            else:
                # Continuous action model (Dual Mixed CQL)
                action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                q1 = self.agent.q1(state_t, action_t).item()
                q2 = self.agent.q2(state_t, action_t).item()
            
            return min(q1, q2)