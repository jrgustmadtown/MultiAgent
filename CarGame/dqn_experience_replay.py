"""
DQN with Experience Replay - Eliminates catastrophic forgetting
"""

import numpy as np
import torch
from Gridworld import Gridworld
from IPython.display import clear_output
import random
from collections import deque
from dqn_models import create_dqn_model
from utils import plot_losses
from config import (
    EXPERIENCE_REPLAY_CONFIG, ACTION_SET, LAYER_SIZES,
    GRID_SIZE, NOISE_FACTOR_TRAIN
)


def train_experience_replay_dqn(config=None):
    """
    Train DQN with experience replay to eliminate catastrophic forgetting
    
    Args:
        config: Dictionary of configuration parameters (uses EXPERIENCE_REPLAY_CONFIG if None)
        
    Returns:
        model: Trained model
        losses: List of training losses
    """
    if config is None:
        config = EXPERIENCE_REPLAY_CONFIG
    
    print("=" * 60)
    print("Training DQN with Experience Replay")
    print("=" * 60)
    print(f"Epochs: {config['epochs']}")
    print(f"Gamma: {config['gamma']}")
    print(f"Epsilon: {config['epsilon']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Memory Size: {config['mem_size']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Max Moves: {config['max_moves']}")
    print(f"Game Mode: {config['mode']}")
    print("=" * 60)
    
    # Create model
    model = create_dqn_model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training parameters
    gamma = config['gamma']
    epsilon = config['epsilon']
    epochs = config['epochs']
    mem_size = config['mem_size']
    batch_size = config['batch_size']
    max_moves = config['max_moves']
    mode = config['mode']
    
    # Experience replay buffer
    replay = deque(maxlen=mem_size)
    losses = []
    
    for i in range(epochs):
        game = Gridworld(size=GRID_SIZE, mode=mode)
        state1_ = game.board.render_np().reshape(1, LAYER_SIZES['l1']) + np.random.rand(1, LAYER_SIZES['l1']) / NOISE_FACTOR_TRAIN
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while status == 1:
            mov += 1
            
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
            done = True if reward > 0 else False
            
            # Store experience
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            # Train on minibatch
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                
                # Prepare batch tensors
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                # Calculate Q-values
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                # Calculate target and loss
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                print(f"Epoch {i}/{epochs}, Loss: {loss.item():.6f}, Replay Size: {len(replay)}")
                clear_output(wait=True)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            # Check if game is over
            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0
    
    print("\nTraining completed!")
    return model, losses


if __name__ == "__main__":
    model, losses = train_experience_replay_dqn()
    plot_losses(losses, title="Experience Replay DQN Training Loss", smooth=True)
