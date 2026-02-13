"""
Utility functions for testing, evaluation, and visualization
"""

import numpy as np
import torch
from matplotlib import pylab as plt
from config import ACTION_SET, MAX_TEST_MOVES, PLOT_CONFIG, RUNNING_MEAN_WINDOW
from Gridworld import Gridworld


def test_model(model, mode='static', display=True, noise_factor=10.0):
    """
    Test the trained model on a single game
    
    Args:
        model: Trained DQN model
        mode: Game mode ('static' or 'random')
        display: Whether to print game states
        noise_factor: Amount of noise to add to state
        
    Returns:
        win: Boolean indicating if the game was won
    """
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / noise_factor
    state = torch.from_numpy(state_).float()
    
    if display:
        print("Initial State:")
        print(test_game.display())
        
    status = 1
    while status == 1:
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = ACTION_SET[action_]
        
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
            
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / noise_factor
        state = torch.from_numpy(state_).float()
        
        if display:
            print(test_game.display())
            
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if i > MAX_TEST_MOVES:
            if display:
                print("Game lost; too many moves.")
            break
    
    win = True if status == 2 else False
    return win


def evaluate_model(model, mode='random', max_games=1000, noise_factor=10.0):
    """
    Evaluate model performance over multiple games
    
    Args:
        model: Trained DQN model
        mode: Game mode ('static' or 'random')
        max_games: Number of games to play
        noise_factor: Amount of noise to add to state
        
    Returns:
        win_perc: Win percentage
    """
    wins = 0
    for i in range(max_games):
        win = test_model(model, mode=mode, display=False, noise_factor=noise_factor)
        if win:
            wins += 1
    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games, wins))
    print("Win percentage: {:.1f}%".format(100.0 * win_perc))
    return win_perc


def running_mean(x, N=None):
    """
    Calculate running mean for smoothing loss plots
    
    Args:
        x: Array of values
        N: Window size (default from config)
        
    Returns:
        y: Smoothed array
    """
    if N is None:
        N = RUNNING_MEAN_WINDOW
    c = x.shape[0] - N
    if c <= 0:
        return x
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv) / N
    return y


def plot_losses(losses, title="Training Loss", smooth=False, save_path=None):
    """
    Plot training losses
    
    Args:
        losses: List or array of loss values
        title: Plot title
        smooth: Whether to apply smoothing
        save_path: If provided, save plot to this file instead of showing
    """
    losses = np.array(losses)
    
    plt.figure(figsize=PLOT_CONFIG['figsize'])
    
    if smooth and len(losses) > RUNNING_MEAN_WINDOW:
        smoothed = running_mean(losses)
        plt.plot(smoothed, label='Smoothed')
        plt.plot(losses, alpha=0.3, label='Raw')
        plt.legend()
    else:
        plt.plot(losses)
    
    plt.xlabel("Iterations", fontsize=PLOT_CONFIG['xlabel_fontsize'])
    plt.ylabel("Loss", fontsize=PLOT_CONFIG['ylabel_fontsize'])
    plt.title(title, fontsize=PLOT_CONFIG['title_fontsize'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def save_model(model, filepath):
    """Save model to file"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """Load model from file"""
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model
