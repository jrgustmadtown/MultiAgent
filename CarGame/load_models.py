"""
Load and test saved multi-agent DQN models
"""

import torch
import sys
sys.path.append('..')
from dqn_models import create_dqn_model
from utils import test_models, evaluate_models
from config import NOISE_FACTOR_TEST, EVAL_GAMES


def load_models(path_a='model_a.pth', path_b='model_b.pth'):
    """
    Load both trained models
    
    Args:
        path_a: Path to Agent A model file
        path_b: Path to Agent B model file
        
    Returns:
        model_a: Loaded Agent A model
        model_b: Loaded Agent B model
    """
    # Create model architectures
    model_a = create_dqn_model()
    model_b = create_dqn_model()
    
    # Load saved weights
    model_a.load_state_dict(torch.load(path_a))
    model_b.load_state_dict(torch.load(path_b))
    
    # Set to evaluation mode
    model_a.eval()
    model_b.eval()
    
    print(f"Models loaded successfully!")
    print(f"Agent A from: {path_a}")
    print(f"Agent B from: {path_b}")
    
    return model_a, model_b


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and test saved multi-agent models')
    parser.add_argument('--model-a', type=str, default='model_a.pth',
                       help='Path to Agent A model file')
    parser.add_argument('--model-b', type=str, default='model_b.pth',
                       help='Path to Agent B model file')
    parser.add_argument('--no-demo', action='store_true',
                       help='Skip demo game')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation')
    parser.add_argument('--games', type=int, default=EVAL_GAMES,
                       help='Number of games for evaluation')
    
    args = parser.parse_args()
    
    # Load models
    model_a, model_b = load_models(args.model_a, args.model_b)
    
    # Play demo game
    if not args.no_demo:
        print("\n" + "="*70)
        print("DEMO GAME")
        print("="*70 + "\n")
        test_models(model_a, model_b, display=True, noise_factor=NOISE_FACTOR_TEST)
    
    # Evaluate
    if not args.no_eval:
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70 + "\n")
        evaluate_models(model_a, model_b, max_games=args.games, 
                       noise_factor=NOISE_FACTOR_TEST)
