"""
TD-Inspired Reward Function for Language Models

This module implements a reward function based on temporal difference (TD) learning
principles, inspired by research on dopamine signaling in contingency learning.

The core concept is to reward the model for accurately predicting future tokens
(prospective learning) rather than relying on retrospective associations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class BeliefStateModel(nn.Module):
    """
    Implements a belief state representation for language contexts.
    
    Inspired by the Belief-State TD model from the research, this module
    maintains a vector of probabilities representing beliefs about the
    current state of the conversation.
    """
    
    def __init__(self, hidden_size, num_belief_states=2):
        """
        Initialize the belief state model.
        
        Args:
            hidden_size: Dimension of the input hidden states
            num_belief_states: Number of possible belief states to track
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_belief_states = num_belief_states
        
        # Projection from hidden states to belief state probabilities
        self.projection = nn.Linear(hidden_size, num_belief_states)
        
        # State transition matrix (learnable)
        self.transition_matrix = nn.Parameter(
            torch.ones(num_belief_states, num_belief_states) / num_belief_states
        )
    
    def forward(self, hidden_states):
        """
        Compute belief state probabilities from hidden states.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            
        Returns:
            Belief state probabilities [batch_size, seq_len, num_belief_states]
        """
        # Project hidden states to logits
        logits = self.projection(hidden_states)
        
        # Convert to probabilities
        belief_probs = F.softmax(logits, dim=-1)
        
        return belief_probs
    
    def update_beliefs(self, current_beliefs, observations=None):
        """
        Update belief states based on transition dynamics and observations.
        
        Args:
            current_beliefs: Current belief state probabilities [batch_size, num_belief_states]
            observations: Optional observation probabilities to incorporate
            
        Returns:
            Updated belief state probabilities [batch_size, num_belief_states]
        """
        # Apply transition dynamics
        updated_beliefs = torch.matmul(current_beliefs, self.transition_matrix)
        
        # Incorporate observations if provided (using Bayes' rule)
        if observations is not None:
            # Normalize to ensure valid probabilities
            updated_beliefs = updated_beliefs * observations
            updated_beliefs = updated_beliefs / (updated_beliefs.sum(dim=-1, keepdim=True) + 1e-10)
            
        return updated_beliefs


class ValueFunction(nn.Module):
    """
    Estimates the value (expected future reward) for a given state.
    
    This is similar to the critic in actor-critic RL methods, but specifically
    tailored to estimate TD errors for language contexts.
    """
    
    def __init__(self, hidden_size, belief_state_size):
        """
        Initialize the value function.
        
        Args:
            hidden_size: Dimension of the input hidden states
            belief_state_size: Dimension of the belief state representation
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size + belief_state_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states, belief_states):
        """
        Compute the value estimate for a given state.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, hidden_size]
            belief_states: Belief state probabilities [batch_size, belief_state_size]
            
        Returns:
            Value estimates [batch_size, 1]
        """
        # Concatenate hidden states and belief states
        combined = torch.cat([hidden_states, belief_states], dim=-1)
        
        # Project to value estimate
        value = self.projection(combined)
        
        return value


class TDRewardModel:
    """
    Implements a TD-inspired reward model for language generation.
    
    This model computes rewards based on TD errors, emphasizing prospective
    prediction accuracy over retrospective associations.
    """
    
    def __init__(
        self, 
        model_name="gpt2", 
        alpha=0.1,  # Weight for retrospective penalty
        beta=0.01,  # Weight for exploration bonus
        gamma=0.99  # Discount factor for future rewards
    ):
        """
        Initialize the TD reward model.
        
        Args:
            model_name: Name of the base language model
            alpha: Weight for retrospective penalty component
            beta: Weight for exploration/novelty bonus
            gamma: Discount factor for future rewards
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.language_model.config.hidden_size
        
        # Initialize belief state model
        self.belief_state_model = BeliefStateModel(hidden_size)
        
        # Initialize value function
        self.value_function = ValueFunction(
            hidden_size, 
            self.belief_state_model.num_belief_states
        )
    
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """
        Create a TDRewardModel from a pretrained language model.
        
        Args:
            model_name: Name of the pretrained model
            **kwargs: Additional arguments for the TDRewardModel constructor
            
        Returns:
            Initialized TDRewardModel
        """
        return cls(model_name=model_name, **kwargs)
    
    def compute_td_reward(self, context, prediction, actual_next_token):
        """
        Compute the TD-inspired reward for a prediction.
        
        Args:
            context: Current context (text or token IDs)
            prediction: Model's prediction (text or token IDs)
            actual_next_token: Actual next token (text or token ID)
            
        Returns:
            Total reward value (scalar)
        """
        # Tokenize inputs if they're not already tokenized
        if isinstance(context, str):
            context = self.tokenizer.encode(context, return_tensors="pt")
        if isinstance(prediction, str):
            prediction = self.tokenizer.encode(prediction, return_tensors="pt")
        if isinstance(actual_next_token, str):
            actual_next_token = self.tokenizer.encode(actual_next_token, return_tensors="pt")[0, 0]
        
        # Debug information
        print(f"Context device: {context.device}")
        print(f"Prediction device: {prediction.device if isinstance(prediction, torch.Tensor) else 'N/A'}")
        print(f"Actual next token device: {actual_next_token.device if isinstance(actual_next_token, torch.Tensor) else 'N/A'}")
        print(f"Language model device: {self.language_model.device}")
        print(f"Belief state model device: {next(self.belief_state_model.parameters()).device}")
        print(f"Value function device: {next(self.value_function.parameters()).device}")
        
        # Ensure all tensors are on the same device
        device = self.language_model.device
        if context.device != device:
            context = context.to(device)
        if isinstance(prediction, torch.Tensor) and prediction.device != device:
            prediction = prediction.to(device)
        if isinstance(actual_next_token, torch.Tensor) and actual_next_token.device != device:
            actual_next_token = actual_next_token.to(device)
        
        # Make sure model components are on the right device
        self.belief_state_model = self.belief_state_model.to(device)
        self.value_function = self.value_function.to(device)
        
        # Make sure actual_next_token has the right shape for concatenation
        if len(actual_next_token.shape) == 1:
            actual_next_token = actual_next_token.unsqueeze(0)
        if len(actual_next_token.shape) == 2 and actual_next_token.shape[0] == 1:
            # This is already in the shape [1, 1]
            pass
        elif len(actual_next_token.shape) == 0:
            # This is a scalar
            actual_next_token = actual_next_token.unsqueeze(0).unsqueeze(0)
        
        # After correcting devices
        print(f"Context device after correction: {context.device}")
        print(f"Prediction device after correction: {prediction.device if isinstance(prediction, torch.Tensor) else 'N/A'}")
        print(f"Actual next token device after correction: {actual_next_token.device if isinstance(actual_next_token, torch.Tensor) else 'N/A'}")
        
        # Get model outputs
        with torch.no_grad():
            # Get hidden states for context
            context_outputs = self.language_model(
                context, output_hidden_states=True, return_dict=True
            )
            context_hidden_states = context_outputs.hidden_states[-1][:, -1]  # Last token, last layer
            
            # Compute current belief states
            current_belief_states = self.belief_state_model(
                context_hidden_states.unsqueeze(1)
            ).squeeze(1)
            
            # Compute current value estimate
            current_value = self.value_function(
                context_hidden_states, current_belief_states
            )
            
            # Get next state (context + actual next token)
            next_context = torch.cat([context, actual_next_token], dim=1)
            next_outputs = self.language_model(
                next_context, output_hidden_states=True, return_dict=True
            )
            next_hidden_states = next_outputs.hidden_states[-1][:, -1]
            
            # Compute next belief states
            next_belief_states = self.belief_state_model(
                next_hidden_states.unsqueeze(1)
            ).squeeze(1)
            
            # Compute next value estimate
            next_value = self.value_function(
                next_hidden_states, next_belief_states
            )
            
            # Compute prediction probabilities
            prediction_outputs = self.language_model(context)
            prediction_probs = F.softmax(prediction_outputs.logits[:, -1], dim=-1)
            
            # Get the probability of the actual next token
            actual_next_token_idx = actual_next_token[0, 0].item()
            actual_next_token_prob = prediction_probs[0, actual_next_token_idx]
            
            # Compute the entropy of the prediction distribution
            prediction_entropy = -torch.sum(
                prediction_probs * torch.log(prediction_probs + 1e-10)
            )
            
            # Compute retrospective probability (P(Context|Next Token))
            # This is an approximation - in a full implementation, would require 
            # more sophisticated calculations
            retrospective_prob = actual_next_token_prob  # Simplified for now
        
        # Compute reward components
        
        # 1. Predictive accuracy reward (log probability of actual next token)
        r_prediction = torch.log(actual_next_token_prob + 1e-10)
        
        # 2. Retrospective penalty (penalize reliance on retrospective alignment)
        r_retrospective = -self.alpha * torch.log(retrospective_prob + 1e-10)
        
        # 3. Exploration/novelty bonus (entropy of prediction distribution)
        r_exploration = self.beta * prediction_entropy
        
        # Immediate reward
        r_immediate = r_prediction + r_retrospective + r_exploration
        
        # Compute TD error: r_t + γV(s_{t+1}) - V(s_t)
        td_error = r_immediate + self.gamma * next_value - current_value
        
        # Total reward is the TD error
        total_reward = td_error.item()
        
        return total_reward


# Helper function to demonstrate usage
def example_td_reward_calculation():
    """
    Example of how to use the TD reward model.
    """
    # Initialize the model
    model = TDRewardModel.from_pretrained("gpt2")
    
    # Example inputs
    context = "The quick brown fox jumps over the"
    prediction = "lazy"
    actual_next_token = "lazy"
    
    # Compute reward
    reward = model.compute_td_reward(context, prediction, actual_next_token)
    
    print(f"Context: {context}")
    print(f"Prediction: {prediction}")
    print(f"Actual next token: {actual_next_token}")
    print(f"TD reward: {reward}")
    
    return reward


if __name__ == "__main__":
    example_td_reward_calculation()
