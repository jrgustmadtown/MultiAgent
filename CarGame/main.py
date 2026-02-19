"""
Main file to run Multi-Agent Minimax DQN training and evaluation
Car Crash Game: Agent A (Pursuer) vs Agent B (Evader)
"""

import argparse
import numpy as np
from minimax_dqn import train_minimax_dqn
from utils import test_models, evaluate_models, plot_losses, save_model, visualize_policy, clear_policy_images
from config import MINIMAX_CONFIG, EVAL_GAMES, NOISE_FACTOR_TEST, GRID_SIZE
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
    parser.add_argument('--visualize-policy', action='store_true',
                       help='Generate policy visualization images for all opponent positions')
    parser.add_argument('--policy-dir', type=str, default='policy_images',
                       help='Directory to save policy visualizations')
    parser.add_argument('--policy-steps', type=int, default=1,
                       help='Number of successive moves to show in policy visualizations (default: 1)')
    parser.add_argument('--clear-policy', action='store_true',
                       help='Clear all policy visualization images and exit')
    
    args = parser.parse_args()
    
    # Handle clear-policy command
    if args.clear_policy:
        clear_policy_images(args.policy_dir)
        return
    
    # Override config if epochs specified
    config = MINIMAX_CONFIG.copy()
    if args.epochs:
        config['epochs'] = args.epochs
    
    print("\n" + "="*70)
    print("MINIMAX DQN - CAR CRASH GAME")
    print("="*70 + "\n")
    
    # Train both agents
    model_a, model_b, losses_a, losses_b, scores_a, scores_b = train_minimax_dqn(config)
    
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
        
        # Always save plot, use default name if not specified
        plot_filename = args.save_plot if args.save_plot else 'training_losses.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to {plot_filename}")
        plt.close()
    
    # Plot scores over epochs
    if not args.no_plot:
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(scores_a, color='red', alpha=0.6, label='Agent A')
        plt.plot(scores_b, color='blue', alpha=0.6, label='Agent B')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.title("Cumulative Scores per Epoch", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        
        # Running average of scores
        plt.subplot(1, 2, 2)
        window = 50
        if len(scores_a) >= window:
            from utils import running_mean
            smooth_a = running_mean(np.array(scores_a), window)
            smooth_b = running_mean(np.array(scores_b), window)
            plt.plot(smooth_a, color='red', linewidth=2, label='Agent A (avg)')
            plt.plot(smooth_b, color='blue', linewidth=2, label='Agent B (avg)')
        else:
            plt.plot(scores_a, color='red', alpha=0.6, label='Agent A')
            plt.plot(scores_b, color='blue', alpha=0.6, label='Agent B')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.title("Scores (Smoothed)", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        score_filename = 'training_scores.png'
        plt.savefig(score_filename, dpi=300, bbox_inches='tight')
        print(f"Score plots saved to {score_filename}")
        plt.close()
    
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
    
    # Generate policy visualizations
    if args.visualize_policy:
        print("\n" + "-"*70)
        print("GENERATING POLICY VISUALIZATIONS")
        print("-"*70)
        total = GRID_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE
        count = 0
        # Generate for all possible state combinations
        for i_a in range(GRID_SIZE):
            for j_a in range(GRID_SIZE):
                for i_b in range(GRID_SIZE):
                    for j_b in range(GRID_SIZE):
                        # Convert grid indices to normalized coordinates
                        pos_a = (i_a / GRID_SIZE, j_a / GRID_SIZE)
                        pos_b = (i_b / GRID_SIZE, j_b / GRID_SIZE)
                        visualize_policy(model_a, model_b, pos_a, pos_b, 
                                       output_dir=args.policy_dir, steps=args.policy_steps)
                        count += 1
                        if count % 50 == 0:
                            print(f"Generated {count}/{total} visualizations...")
        print(f"All {total} policy visualizations saved to {args.policy_dir}/")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # If running without arguments, default to demo mode with fewer epochs
    import sys
    if len(sys.argv) == 1:
        print("Running with default settings: Minimax DQN with demo")
        print("Use --help to see all options\n")
        sys.argv.extend(['--demo', '--epochs', '500'])
    
    main()
