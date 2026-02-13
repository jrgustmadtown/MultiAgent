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


def visualize_policy(model_a, model_b, opponent_pos, output_dir='policy_images'):
    """
    Create policy visualization images showing best move for each position
    
    Args:
        model_a: Trained model for agent A
        model_b: Trained model for agent B
        opponent_pos: Fixed opponent position (row, col) to visualize policy against
        output_dir: Directory to save images
    """
    import os
    from CarBoard import CarBoard
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Action to arrow direction mapping
    action_arrows = {
        0: (0, -0.4),    # up: arrow points up
        1: (0, 0.4),     # down: arrow points down
        2: (-0.4, 0),    # left: arrow points left
        3: (0.4, 0)      # right: arrow points right
    }
    
    # Create separate visualizations for each agent
    for agent, model in [('A', model_a), ('B', model_b)]:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for i in range(GRID_SIZE + 1):
            ax.axhline(i, color='gray', linewidth=0.5)
            ax.axvline(i, color='gray', linewidth=0.5)
        
        # For each position on the grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Create a temporary game state
                board = CarBoard(size=GRID_SIZE)
                
                if agent == 'A':
                    board.car_a_pos = (i, j)
                    board.car_b_pos = opponent_pos
                    car_pos = (i, j)
                else:
                    board.car_a_pos = opponent_pos
                    board.car_b_pos = (i, j)
                    car_pos = (i, j)
                
                # Get state and predict best action
                state = torch.from_numpy(board.get_state_vector()).float()
                qvals = model(state).data.numpy()
                
                # Get valid actions
                from CarGame import CarGame, addTuple
                temp_game = CarGame(size=GRID_SIZE)
                temp_game.board = board
                valid_actions = temp_game.get_valid_actions(agent)
                
                # Find best valid action
                valid_qvals = [(a, qvals[a]) for a in valid_actions]
                best_action = max(valid_qvals, key=lambda x: x[1])[0]
                
                # Draw arrow for best action
                dx, dy = action_arrows[best_action]
                # Convert (row, col) to (x, y) for plotting: x=col, y=row (inverted)
                x = j + 0.5
                y = GRID_SIZE - i - 0.5  # Invert y-axis
                
                color = 'red' if agent == 'A' else 'blue'
                ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1, 
                        fc=color, ec=color, linewidth=2)
        
        # Mark opponent position
        opp_x = opponent_pos[1] + 0.5
        opp_y = GRID_SIZE - opponent_pos[0] - 0.5
        opp_color = 'blue' if agent == 'A' else 'red'
        opp_label = 'B' if agent == 'A' else 'A'
        ax.plot(opp_x, opp_y, 'o', color=opp_color, markersize=20, 
               markeredgecolor='black', markeredgewidth=2)
        ax.text(opp_x, opp_y, opp_label, ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white')
        
        # Formatting
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(f'Agent {agent} Policy (Opponent at {opponent_pos})', fontsize=16)
        ax.invert_yaxis()  # Row 0 at top
        
        # Save figure
        filename = f'{output_dir}/agent_{agent}_vs_{opponent_pos[0]}_{opponent_pos[1]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved policy visualization: {filename}")
        plt.close()


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
