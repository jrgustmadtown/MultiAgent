"""
Main file to run Multi-Agent Minimax DQN training and evaluation
Car Crash Game: Agent A (Pursuer) vs Agent B (Evader)
"""

import argparse
from minimax_dqn import train_minimax_dqn
from utils import test_models, evaluate_models, plot_losses, save_model
from config import MINIMAX_CONFIG, EVAL_GAMES, NOISE_FACTOR_TEST
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Multi-Agent DQN for Car Crash Game')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting losses')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--save-a', type=str, default=None,
                       help='Path to save Agent A model (e.g., model_a.pth)')
    parser.add_argument('--save-b', type=str, default=None,
                       help='Path to save Agent B model (e.g., model_b.pth)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save combined plot (e.g., losses.png)')
    parser.add_argument('--demo', action='store_true',
                       help='Show a demo game after training')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    
    args = parser.parse_args()
    
    # Override config if epochs specified
    config = MINIMAX_CONFIG.copy()
    if args.epochs:
        config['epochs'] = args.epochs
    
    print("\n" + "="*70)
    print("MINIMAX DQN - CAR CRASH GAME")
    print("="*70 + "\n")
    
    # Train both agents
    model_a, model_b, losses_a, losses_b = train_minimax_dqn(config)
    
    # Plot losses
    if not args.no_plot:
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses_a)
        plt.title("Agent A (Pursuer) Training Loss", fontsize=16)
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(losses_b)
        plt.title("Agent B (Evader) Training Loss", fontsize=16)
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {args.save_plot}")
            plt.close()
        else:
            plt.show()
    
    # Evaluate the models
    if not args.no_eval:
        print("\n" + "-"*70)
        print("EVALUATION")
        print("-"*70)
        evaluate_models(model_a, model_b, max_games=EVAL_GAMES, 
                       noise_factor=NOISE_FACTOR_TEST)
    
    # Show demo game
    if args.demo:
        print("\n" + "-"*70)
        print("DEMO GAME")
        print("-"*70)
        test_models(model_a, model_b, display=True, 
                   noise_factor=NOISE_FACTOR_TEST)
    
    # Save models
    if args.save_a:
        save_model(model_a, args.save_a)
    if args.save_b:
        save_model(model_b, args.save_b)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # If running without arguments, default to demo mode with fewer epochs
    import sys
    if len(sys.argv) == 1:
        print("Running with default settings: Minimax DQN with demo")
        print("Use --help to see all options\n")
        sys.argv.extend(['--demo', '--epochs', '500'])
    
    main()
        print("Running with default settings: Target Network DQN with demo")
        print("Use --help to see all options\n")
        sys.argv.extend(['--model', 'target', '--demo'])
    
    main()
