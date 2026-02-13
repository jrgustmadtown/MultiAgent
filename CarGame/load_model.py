"""
Load and test a saved DQN model
"""

import torch
from dqn_models import create_dqn_model
from utils import test_model, evaluate_model
from config import NOISE_FACTOR_TEST

# Load the saved model
model = create_dqn_model()
model.load_state_dict(torch.load('model.pth'))
model.eval()

print("Model loaded successfully!")
print("\nPlaying a demo game:")
test_model(model, mode='static', display=True, noise_factor=NOISE_FACTOR_TEST)

print("\nEvaluating on 100 random games:")
evaluate_model(model, mode='random', max_games=100, noise_factor=NOISE_FACTOR_TEST)
