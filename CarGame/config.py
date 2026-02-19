"""
Configuration file for DQN Car Crash Game (Multi-Agent)
Modify parameters here to experiment with different settings
"""

# ============================================================================
# GAME SETTINGS
# ============================================================================
GRID_SIZE = 5  # Must be odd and >= 3
MAX_TURNS = 50  # Maximum turns per game
CRASH_REWARD = 10  # Bonus for A, penalty for B when crash occurs (adjusted for normalized points)

ACTION_SET = {
    0: 'u',  # up
    1: 'd',  # down
    2: 'l',  # left
    3: 'r',  # right
}

# ============================================================================
# NETWORK ARCHITECTURE (calculated from GRID_SIZE)
# ============================================================================
LAYER_SIZES = {
    'l1': GRID_SIZE * GRID_SIZE * 2,  # Input: grid x 2 channels (auto-calculated)
    'l2': 128,  # First hidden layer
    'l3': 128,  # Second hidden layer
    'l4': 4     # Output layer (4 actions: up, down, left, right)
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Minimax DQN Configuration (Multi-Agent)
MINIMAX_CONFIG = {
    'epochs': 3000,
    'gamma': 0.95,          # Discount factor (higher = more long-term planning)
    'epsilon': 0.2,         # Exploration rate (lower = more exploitation)
    'learning_rate': 5e-4,  # Lower LR for more stable learning
    'mem_size': 2000,       # Larger memory for better experience diversity
    'batch_size': 256,      # Larger batches for more stable gradients
    'sync_freq': 250,       # Sync more frequently for better stability
}

# Vanilla DQN (kept for reference, may not work well for multi-agent)
VANILLA_CONFIG = {
    'epochs': 100,
    'gamma': 0.9,           # Discount factor
    'epsilon_start': 1.0,   # Initial exploration rate
    'epsilon_end': 0.1,     # Minimum exploration rate
    'learning_rate': 1e-3,
}

# Experience Replay DQN (kept for reference)
EXPERIENCE_REPLAY_CONFIG = {
    'epochs': 3000,
    'gamma': 0.9,
    'epsilon': 0.3,         # Fixed exploration rate
    'learning_rate': 1e-3,
    'mem_size': 1000,       # Experience replay memory size
    'batch_size': 200,      # Minibatch size
}

# Target Network DQN (kept for reference)
TARGET_NETWORK_CONFIG = {
    'epochs': 3000,
    'gamma': 0.9,
    'epsilon': 0.3,
    'learning_rate': 1e-3,
    'mem_size': 1000,
    'batch_size': 200,
    'sync_freq': 500,       # How often to sync target network
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVAL_GAMES = 100            # Number of games for evaluation
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
