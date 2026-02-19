"""
Utility functions for testing, evaluation, and visualization (Multi-Agent)
"""

import numpy as np
import torch
import sys
sys.path.append('..')
from matplotlib import pylab as plt
from config import ACTION_SET, PLOT_CONFIG, RUNNING_MEAN_WINDOW, GRID_SIZE, MAX_TURNS, LAYER_SIZES
from CarGame import CarGame


def test_models(model_a, model_b, display=True, noise_factor=10.0):
    """
    Test both trained models on a single game
    
    Args:
        model_a: Trained DQN model for agent A (pursuer)
        model_b: Trained DQN model for agent B (evader)
        display: Whether to print game states
        noise_factor: Amount of noise to add to state
        
    Returns:
        crashed: Boolean indicating if A caught B
        turns: Number of turns played
        score_a: Final score for A
        score_b: Final score for B
    """
    game = CarGame(size=GRID_SIZE, max_turns=MAX_TURNS)
    state_ = game.get_state() + np.random.rand(LAYER_SIZES['l1']) / noise_factor
    state = torch.from_numpy(state_).float()
    
    if display:
        print("Initial State:")
        print(game.display())
        print(f"Car A: {game.board.car_a_pos}, Car B: {game.board.car_b_pos}")
        print()
    
    turn = 0
    while not game.game_over:
        # Get valid actions
        valid_actions_a = game.get_valid_actions('A')
        valid_actions_b = game.get_valid_actions('B')
        
        # Agent A selects action (greedy, only from valid actions)
        qval_a = model_a(state)
        qval_a_ = qval_a.data.numpy()
        valid_qvals_a = [(a, qval_a_[a]) for a in valid_actions_a]
        action_a = max(valid_qvals_a, key=lambda x: x[1])[0]
        
        # Agent B selects action (greedy, only from valid actions)
        qval_b = model_b(state)
        qval_b_ = qval_b.data.numpy()
        valid_qvals_b = [(a, qval_b_[a]) for a in valid_actions_b]
        action_b = max(valid_qvals_b, key=lambda x: x[1])[0]
        
        if display:
            print(f'Turn {turn}: A moves {list(ACTION_SET.values())[action_a]}, B moves {list(ACTION_SET.values())[action_b]}')
        
        # Execute both actions
        reward_a, reward_b, game_over = game.executeRound(action_a, action_b)
        
        # Get next state
        state_ = game.get_state() + np.random.rand(LAYER_SIZES['l1']) / noise_factor
        state = torch.from_numpy(state_).float()
        
        if display:
            print(game.display())
            print(f"Car A: {game.board.car_a_pos}, Car B: {game.board.car_b_pos}")
            print(f"Rewards: A={reward_a}, B={reward_b}")
            print()
        
        turn += 1
        
        if game_over:
            break
    
    info = game.get_info()
    
    if display:
        if info['crashed']:
            print(f"CRASH! Agent A caught Agent B in {turn} turns!")
        else:
            print(f"Agent B survived for {turn} turns!")
        print(f"Final Scores: A={info['score_a']}, B={info['score_b']}")
    
    return info['crashed'], turn, info['score_a'], info['score_b']


def evaluate_models(model_a, model_b, max_games=100, noise_factor=10.0):
    """
    Evaluate both models over multiple games
    
    Args:
        model_a: Trained DQN model for agent A
        model_b: Trained DQN model for agent B
        max_games: Number of games to play
        noise_factor: Amount of noise to add to state
        
    Returns:
        stats: Dictionary of evaluation statistics
    """
    crashes = 0
    total_turns = 0
    scores_a = []
    scores_b = []
    
    for i in range(max_games):
        crashed, turns, score_a, score_b = test_models(
            model_a, model_b, display=False, noise_factor=noise_factor
        )
        if crashed:
            crashes += 1
        total_turns += turns
        scores_a.append(score_a)
        scores_b.append(score_b)
    
    crash_rate = (crashes / max_games) * 100
    avg_turns = total_turns / max_games
    avg_score_a = np.mean(scores_a)
    avg_score_b = np.mean(scores_b)
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Games played: {max_games}")
    print(f"Crashes (A wins): {crashes} ({crash_rate:.1f}%)")
    print(f"Survivals (B wins): {max_games - crashes} ({100 - crash_rate:.1f}%)")
    print(f"Average game length: {avg_turns:.1f} turns")
    print(f"Average score A: {avg_score_a:.1f}")
    print(f"Average score B: {avg_score_b:.1f}")
    print("=" * 60)
    
    return {
        'crash_rate': crash_rate,
        'avg_turns': avg_turns,
        'avg_score_a': avg_score_a,
        'avg_score_b': avg_score_b,
        'crashes': crashes,
        'survivals': max_games - crashes
    }


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


def visualize_policy(model_a, model_b, car_a_pos, car_b_pos, output_dir='policy_images', steps=1):
    """
    Create policy visualization showing trajectory of best moves for a specific state
    
    Args:
        model_a: Trained model for agent A
        model_b: Trained model for agent B
        car_a_pos: Starting position of car A (row, col)
        car_b_pos: Starting position of car B (row, col)
        output_dir: Directory to save images
        steps: Number of successive moves to show (default 1 = immediate best move only)
    """
    import os
    from CarBoard import CarBoard
    from CarGame import CarGame, addTuple
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color='gray', linewidth=1)
        ax.axvline(i, color='gray', linewidth=1)
    
    # Simulate trajectory
    current_a = car_a_pos
    current_b = car_b_pos
    
    for step in range(steps):
        # Create game state
        board = CarBoard(size=GRID_SIZE)
        board.car_a_pos = current_a
        board.car_b_pos = current_b
        
        # Get state vector
        state = torch.from_numpy(board.get_state_vector()).float()
        
        # Create temporary game to get valid actions
        temp_game = CarGame(size=GRID_SIZE)
        temp_game.board = board
        
        # Get valid actions for both agents
        valid_actions_a = temp_game.get_valid_actions('A')
        valid_actions_b = temp_game.get_valid_actions('B')
        
        # Agent A's best action
        qvals_a = model_a(state).data.numpy()
        valid_qvals_a = [(a, qvals_a[a]) for a in valid_actions_a]
        best_action_a = max(valid_qvals_a, key=lambda x: x[1])[0]
        
        # Agent B's best action
        qvals_b = model_b(state).data.numpy()
        valid_qvals_b = [(a, qvals_b[a]) for a in valid_actions_b]
        best_action_b = max(valid_qvals_b, key=lambda x: x[1])[0]
        
        # Calculate alpha (fade older moves)
        alpha = 0.3 + 0.7 * (step + 1) / steps
        
        # Calculate next positions
        delta_a = temp_game.ACTIONS[best_action_a]
        delta_b = temp_game.ACTIONS[best_action_b]
        next_a = addTuple(current_a, delta_a)
        next_b = addTuple(current_b, delta_b)
        
        # Convert normalized positions to grid coordinates for plotting
        # Positions are in [0,1], grid is GRID_SIZE x GRID_SIZE
        grid_curr_a = (current_a[0] * GRID_SIZE, current_a[1] * GRID_SIZE)
        grid_curr_b = (current_b[0] * GRID_SIZE, current_b[1] * GRID_SIZE)
        grid_next_a = (next_a[0] * GRID_SIZE, next_a[1] * GRID_SIZE)
        grid_next_b = (next_b[0] * GRID_SIZE, next_b[1] * GRID_SIZE)
        
        # Agent A arrow: from current center to next center (with small offset)
        offset_a = 0.08  # Offset right and down
        x_a_start = grid_curr_a[1] + offset_a
        y_a_start = grid_curr_a[0] + offset_a
        x_a_end = grid_next_a[1] + offset_a
        y_a_end = grid_next_a[0] + offset_a
        dx_a = x_a_end - x_a_start
        dy_a = y_a_end - y_a_start
        
        ax.arrow(x_a_start, y_a_start, dx_a, dy_a, 
                head_width=0.15, head_length=0.12, 
                fc='red', ec='red', linewidth=2.5, alpha=alpha)
        
        # Agent B arrow: from current center to next center (with small offset)
        offset_b = -0.08  # Offset left and up
        x_b_start = grid_curr_b[1] + offset_b
        y_b_start = grid_curr_b[0] + offset_b
        x_b_end = grid_next_b[1] + offset_b
        y_b_end = grid_next_b[0] + offset_b
        dx_b = x_b_end - x_b_start
        dy_b = y_b_end - y_b_start
        
        ax.arrow(x_b_start, y_b_start, dx_b, dy_b, 
                head_width=0.15, head_length=0.12, 
                fc='blue', ec='blue', linewidth=2.5, alpha=alpha)
        
        # Update positions for next step
        current_a = next_a
        current_b = next_b
        
        # Stop if they crash
        if current_a == current_b:
            break
    
    # Formatting
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    title = f'Trajectory: A from {car_a_pos}, B from {car_b_pos}'
    if steps > 1:
        title += f' ({steps} steps)'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.invert_yaxis()  # Row 0 at top
    
    # Save figure
    filename = f'{output_dir}/state_A{car_a_pos[0]}{car_a_pos[1]}_B{car_b_pos[0]}{car_b_pos[1]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def clear_policy_images(directory='policy_images'):
    """Clear all policy visualization images"""
    import os
    import shutil
    
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleared all files in {directory}/")
    else:
        print(f"Directory {directory}/ does not exist")


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
