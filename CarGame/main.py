"""
Main file to run DQN training and evaluation
"""

import argparse
from dqn_vanilla import train_vanilla_dqn
from dqn_experience_replay import train_experience_replay_dqn
from dqn_target_network import train_target_network_dqn
from utils import test_model, evaluate_model, plot_losses, save_model
from config import (
    VANILLA_CONFIG, EXPERIENCE_REPLAY_CONFIG, TARGET_NETWORK_CONFIG,
    EVAL_GAMES, NOISE_FACTOR_TEST
)


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate DQN models for GridWorld')
    parser.add_argument('--model', type=str, default='target', 
                       choices=['vanilla', 'replay', 'target', 'all'],
                       help='Which model to train: vanilla, replay, target, or all')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting losses')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save trained model')
    parser.add_argument('--demo', action='store_true',
                       help='Show a demo game after training')
    
    args = parser.parse_args()
    
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['vanilla', 'replay', 'target']
    else:
        models_to_train = [args.model]
    
    for model_type in models_to_train:
        print("\n" + "="*70)
        print(f"TRAINING: {model_type.upper()} DQN")
        print("="*70 + "\n")
        
        # Train the model
        if model_type == 'vanilla':
            model, losses = train_vanilla_dqn()
            config = VANILLA_CONFIG
            
        elif model_type == 'replay':
            model, losses = train_experience_replay_dqn()
            config = EXPERIENCE_REPLAY_CONFIG
            
        elif model_type == 'target':
            model, losses = train_target_network_dqn()
            config = TARGET_NETWORK_CONFIG
        
        # Plot losses
        if not args.no_plot:
            smooth = (model_type != 'vanilla')  # Smooth for longer training runs
            plot_losses(losses, title=f"{model_type.title()} DQN Training Loss", smooth=smooth)
        
        # Evaluate the model
        if not args.no_eval:
            print("\n" + "-"*70)
            print("EVALUATION")
            print("-"*70)
            evaluate_model(model, mode=config['mode'], max_games=EVAL_GAMES, 
                         noise_factor=NOISE_FACTOR_TEST)
        
        # Show demo game
        if args.demo:
            print("\n" + "-"*70)
            print("DEMO GAME")
            print("-"*70)
            test_model(model, mode=config['mode'], display=True, 
                      noise_factor=NOISE_FACTOR_TEST)
        
        # Save model
        if args.save:
            save_path = args.save if len(models_to_train) == 1 else f"{args.save}_{model_type}"
            save_model(model, save_path)
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # If running without arguments, default to target network with demo
    import sys
    if len(sys.argv) == 1:
        print("Running with default settings: Target Network DQN with demo")
        print("Use --help to see all options\n")
        sys.argv.extend(['--model', 'target', '--demo'])
    
    main()
