"""
Vanilla DQN Training - Basic Deep Q-Learning without experience replay
"""

import numpy as np
import torch
from Gridworld import Gridworld
from IPython.display import clear_output
import random
from dqn_models import create_dqn_model
from utils import plot_losses
from config import (
    VANILLA_CONFIG, ACTION_SET, LAYER_SIZES, 
    GRID_SIZE, NOISE_FACTOR_TRAIN
)


def train_vanilla_dqn(config=None):
    """
    Train a basic DQN model without experience replay
    
    Args:
        config: Dictionary of configuration parameters (uses VANILLA_CONFIG if None)
        
    Returns:
        model: Trained model
        losses: List of training losses
    """
    if config is None:
        config = VANILLA_CONFIG
    
    print("=" * 60)
    print("Training Vanilla DQN Model")
    print("=" * 60)
    print(f"Epochs: {config['epochs']}")
    print(f"Gamma: {config['gamma']}")
    print(f"Epsilon: {config['epsilon_start']} -> {config['epsilon_end']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Game Mode: {config['mode']}")
    print("=" * 60)
    
    # Create model
    model = create_dqn_model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training parameters
    gamma = config['gamma']
    epsilon = config['epsilon_start']
    epsilon_decay = (config['epsilon_start'] - config['epsilon_end']) / config['epochs']
    epochs = config['epochs']
    mode = config['mode']
    
    losses = []
    
    for i in range(epochs):
        game = Gridworld(size=GRID_SIZE, mode=mode)
        state_ = game.board.render_np().reshape(1, LAYER_SIZES['l1']) + np.random.rand(1, LAYER_SIZES['l1']) / NOISE_FACTOR_TRAIN
        state1 = torch.from_numpy(state_).float()
        status = 1
        
        while status == 1:
            # Select action using epsilon-greedy
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            # Execute action
            action = ACTION_SET[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1, LAYER_SIZES['l1']) + np.random.rand(1, LAYER_SIZES['l1']) / NOISE_FACTOR_TRAIN
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            
            # Calculate target Q-value
            with torch.no_grad():
                newQ = model(state2.reshape(1, LAYER_SIZES['l1']))
            maxQ = torch.max(newQ)
            
            if reward == -1:
                Y = reward + (gamma * maxQ)
            else:
                Y = reward
                
            # Update model
            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action_]
            loss = loss_fn(X, Y)
            
            print(f"Epoch {i}/{epochs}, Loss: {loss.item():.6f}, Epsilon: {epsilon:.3f}")
            clear_output(wait=True)
            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            
            if reward != -1:
                status = 0
        
        # Decay epsilon
        if epsilon > config['epsilon_end']:
            epsilon -= epsilon_decay
    
    print("\nTraining completed!")
    return model, losses


if __name__ == "__main__":
    model, losses = train_vanilla_dqn()
    plot_losses(losses, title="Vanilla DQN Training Loss")
