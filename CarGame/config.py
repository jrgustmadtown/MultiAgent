"""
Configuration file for DQN GridWorld
Modify parameters here to experiment with different settings
"""

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================
LAYER_SIZES = {
    'l1': 64,   # Input layer size (4x4 grid = 64 cells)
    'l2': 150,  # First hidden layer
    'l3': 100,  # Second hidden layer
    'l4': 4     # Output layer (4 actions: up, down, left, right)
}

# ============================================================================
# GAME SETTINGS
# ============================================================================
GRID_SIZE = 4
ACTION_SET = {
    0: 'u',  # up
    1: 'd',  # down
    2: 'l',  # left
    3: 'r',  # right
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Vanilla DQN
VANILLA_CONFIG = {
    'epochs': 100,
    'gamma': 0.9,           # Discount factor
    'epsilon_start': 1.0,   # Initial exploration rate
    'epsilon_end': 0.1,     # Minimum exploration rate
    'learning_rate': 1e-3,
    'mode': 'static'        # Game mode: 'static' or 'random'
}

# Experience Replay DQN
EXPERIENCE_REPLAY_CONFIG = {
    'epochs': 5000,
    'gamma': 0.9,
    'epsilon': 0.3,         # Fixed exploration rate
    'learning_rate': 1e-3,
    'mem_size': 1000,       # Experience replay memory size
    'batch_size': 200,      # Minibatch size
    'max_moves': 50,        # Max moves before game over
    'mode': 'random'
}

# Target Network DQN
TARGET_NETWORK_CONFIG = {
    'epochs': 5000,
    'gamma': 0.9,
    'epsilon': 0.3,
    'learning_rate': 1e-3,
    'mem_size': 1000,
    'batch_size': 200,
    'max_moves': 50,
    'sync_freq': 500,       # How often to sync target network
    'mode': 'random'
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVAL_GAMES = 1000           # Number of games for evaluation
MAX_TEST_MOVES = 15         # Max moves during testing
NOISE_FACTOR_TRAIN = 100.0  # Noise added to state during training
NOISE_FACTOR_TEST = 10.0    # Noise added to state during testing

# ============================================================================
# VISUALIZATION
# ============================================================================
PLOT_CONFIG = {
    'figsize': (10, 7),
    'xlabel_fontsize': 22,
    'ylabel_fontsize': 22,
    'title_fontsize': 16
}
RUNNING_MEAN_WINDOW = 50    # Window size for smoothing loss plots
