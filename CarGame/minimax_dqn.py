"""
Minimax DQN Training - Multi-Agent Adversarial Learning
Car A (pursuer) vs Car B (evader)
"""

import numpy as np
import torch
import sys
sys.path.append('..')
from CarGame import CarGame
from IPython.display import clear_output
import random
from collections import deque
from dqn_models import create_dqn_model, create_target_network, sync_target_network
from utils import plot_losses
from config import (
    MINIMAX_CONFIG, ACTION_SET, LAYER_SIZES,
    GRID_SIZE, MAX_TURNS, NOISE_FACTOR_TRAIN
)


def train_minimax_dqn(config=None):
    """
    Train two DQN agents adversarially using minimax principle
    - Agent A maximizes: reward_A - reward_B (pursuer)
    - Agent B maximizes: reward_B - reward_A (evader)
    
    Args:
        config: Dictionary of configuration parameters (uses MINIMAX_CONFIG if None)
        
    Returns:
        model_a: Trained model for agent A
        model_b: Trained model for agent B
        losses_a: List of training losses for A
        losses_b: List of training losses for B
    """
    if config is None:
        config = MINIMAX_CONFIG
    
    print("=" * 70)
    print("Training Minimax DQN - Multi-Agent Car Crash Game")
    print("=" * 70)
    print(f"Epochs: {config['epochs']}")
    print(f"Gamma: {config['gamma']}")
    print(f"Epsilon: {config['epsilon']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Memory Size: {config['mem_size']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Sync Frequency: {config['sync_freq']}")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Max Turns: {MAX_TURNS}")
    print("=" * 70)
    
    # Create models for both agents
    model_a = create_dqn_model()
    model_b = create_dqn_model()
    
    # Create target networks
    target_a = create_target_network(model_a)
    target_b = create_target_network(model_b)
    
    # Loss functions and optimizers
    loss_fn = torch.nn.MSELoss()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=config['learning_rate'])
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=config['learning_rate'])
    
    # Training parameters
    gamma = config['gamma']
    epsilon = config['epsilon']
    epochs = config['epochs']
    mem_size = config['mem_size']
    batch_size = config['batch_size']
    sync_freq = config['sync_freq']
    
    # Experience replay buffers (separate for each agent)
    replay_a = deque(maxlen=mem_size)
    replay_b = deque(maxlen=mem_size)
    
    losses_a = []
    losses_b = []
    j = 0  # Global step counter
    
    for i in range(epochs):
        game = CarGame(size=GRID_SIZE, max_turns=MAX_TURNS)
        state_ = game.get_state() + np.random.rand(LAYER_SIZES['l1']) / NOISE_FACTOR_TRAIN
        state = torch.from_numpy(state_).float()
        game_over = False
        
        while not game_over:
            j += 1
            
            # Get valid actions for both agents
            valid_actions_a = game.get_valid_actions('A')
            valid_actions_b = game.get_valid_actions('B')
            
            # Agent A selects action (epsilon-greedy, only from valid actions)
            qval_a = model_a(state)
            qval_a_ = qval_a.data.numpy()
            if random.random() < epsilon:
                action_a = random.choice(valid_actions_a)
            else:
                # Choose best action among valid ones
                valid_qvals = [(a, qval_a_[a]) for a in valid_actions_a]
                action_a = max(valid_qvals, key=lambda x: x[1])[0]
            
            # Agent B selects action (epsilon-greedy, only from valid actions)
            qval_b = model_b(state)
            qval_b_ = qval_b.data.numpy()
            if random.random() < epsilon:
                action_b = random.choice(valid_actions_b)
            else:
                # Choose best action among valid ones
                valid_qvals = [(a, qval_b_[a]) for a in valid_actions_b]
                action_b = max(valid_qvals, key=lambda x: x[1])[0]
            
            # Execute both actions simultaneously
            reward_a, reward_b, game_over = game.executeRound(action_a, action_b)
            
            # Get next state
            state2_ = game.get_state() + np.random.rand(LAYER_SIZES['l1']) / NOISE_FACTOR_TRAIN
            state2 = torch.from_numpy(state2_).float()
            
            # Store experiences (separate buffers)
            # Each agent learns from their own rewards
            # Crash bonus/penalty creates the adversarial dynamic
            exp_a = (state, action_a, reward_a, state2, game_over)
            exp_b = (state, action_b, reward_b, state2, game_over)
            
            replay_a.append(exp_a)
            replay_b.append(exp_b)
            
            state = state2
            
            # Train Agent A
            if len(replay_a) > batch_size:
                minibatch = random.sample(replay_a, batch_size)
                
                state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = model_a(state1_batch)
                with torch.no_grad():
                    Q2 = target_a(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss_a = loss_fn(X, Y.detach())
                
                optimizer_a.zero_grad()
                loss_a.backward()
                losses_a.append(loss_a.item())
                optimizer_a.step()
            
            # Train Agent B
            if len(replay_b) > batch_size:
                minibatch = random.sample(replay_b, batch_size)
                
                state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = model_b(state1_batch)
                with torch.no_grad():
                    Q2 = target_b(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss_b = loss_fn(X, Y.detach())
                
                optimizer_b.zero_grad()
                loss_b.backward()
                losses_b.append(loss_b.item())
                optimizer_b.step()
            
            # Display progress
            if j % 100 == 0 and len(losses_a) > 0 and len(losses_b) > 0:
                print(f"Epoch {i}/{epochs}, Step {j}, Loss_A: {losses_a[-1]:.6f}, Loss_B: {losses_b[-1]:.6f}, Replay: {len(replay_a)}, Score_A: {game.score_a:.1f}, Score_B: {game.score_b:.1f}")
            
            # Sync target networks periodically
            if j % sync_freq == 0:
                sync_target_network(model_a, target_a)
                sync_target_network(model_b, target_b)
                print(f"Target networks synchronized at step {j}")
    
    print("\nTraining completed!")
    return model_a, model_b, losses_a, losses_b


if __name__ == "__main__":
    model_a, model_b, losses_a, losses_b = train_minimax_dqn()
    
    # Plot losses for both agents
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_a)
    plt.title("Agent A (Pursuer) Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_b)
    plt.title("Agent B (Evader) Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.show()
