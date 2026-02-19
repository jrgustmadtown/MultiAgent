from CarBoard import CarBoard, addTuple
import numpy as np
import sys
sys.path.append('.')
from config import CRASH_REWARD

class CarGame:
    """
    Two-player car crash game with normalized coordinates
    - Car A (pursuer) tries to crash into Car B
    - Car B (evader) tries to avoid Car A
    - Both move simultaneously
    - Movement scaled by 1/grid_size per turn
    - Crash gives A +CRASH_REWARD, B -CRASH_REWARD
    """
    
    def __init__(self, size=5, max_turns=50):
        if size < 3 or size % 2 == 0:
            raise ValueError("Size must be odd and >= 3")
        
        self.board = CarBoard(size=size)
        self.size = size
        self.max_turns = max_turns
        self.current_turn = 0
        self.game_over = False
        self.crashed = False
        
        # Action deltas scaled to normalized coordinates
        step = 1.0 / size
        self.ACTIONS = {
            0: (-step, 0.0),  # up
            1: (step, 0.0),   # down
            2: (0.0, -step),  # left
            3: (0.0, step)    # right
        }
        if size < 3 or size % 2 == 0:
            raise ValueError("Size must be odd and >= 3")
        
        self.board = CarBoard(size=size)
        self.size = size
        self.max_turns = max_turns
        self.current_turn = 0
        self.game_over = False
        self.crashed = False
        
        # Track cumulative scores
        self.score_a = 0
        self.score_b = 0
    
    def get_valid_actions(self, car='A'):
        """
        Get list of valid action indices for a car at its current position
        With normalized coordinates and clamping, filter out moves that don't change position
        Returns list of action indices (0=up, 1=down, 2=left, 3=right)
        """
        pos = self.board.car_a_pos if car == 'A' else self.board.car_b_pos
        valid = []
        
        for action_idx, delta in self.ACTIONS.items():
            new_pos = addTuple(pos, delta)
            # Clamp the new position
            clamped_pos = (max(0.0, min(1.0, new_pos[0])), max(0.0, min(1.0, new_pos[1])))
            # Only add if the clamped position is different from current (i.e., move is effective)
            if clamped_pos != pos:
                valid.append(action_idx)
        
        # If no valid moves (shouldn't happen except at exact corners), allow all
        if not valid:
            valid = list(self.ACTIONS.keys())
        
        return valid
    
    def executeRound(self, action_a, action_b):
        """
        Execute one round: both players move simultaneously
        Returns: (reward_a, reward_b, game_over)
        Zero-sum game: reward_a + reward_b = 0 always
        Note: Invalid actions should never be passed to this method
        """
        if self.game_over:
            return 0, 0, True
        
        # Get position deltas from actions
        delta_a = self.ACTIONS.get(action_a, (0, 0))
        delta_b = self.ACTIONS.get(action_b, (0, 0))
        
        # Calculate new positions
        new_pos_a = addTuple(self.board.car_a_pos, delta_a)
        new_pos_b = addTuple(self.board.car_b_pos, delta_b)
        
        # Move cars (positions should always be valid)
        self.board.move_car('A', new_pos_a)
        self.board.move_car('B', new_pos_b)
        
        # Calculate rewards (points from squares)
        points_a = self.board.get_square_points(self.board.car_a_pos)
        points_b = self.board.get_square_points(self.board.car_b_pos)
        
        # Zero-sum rewards: your points - opponent's points
        reward_a = points_a - points_b
        reward_b = points_b - points_a
        
        # Check for crash
        if self.board.is_crash():
            self.crashed = True
            self.game_over = True
            reward_a += CRASH_REWARD  # A gets bonus for crashing
            reward_b -= CRASH_REWARD  # B gets penalty for being caught
        
        # Update cumulative scores with zero-sum rewards
        self.score_a += reward_a
        self.score_b += reward_b
        
        # Check for max turns
        self.current_turn += 1
        if self.current_turn >= self.max_turns:
            self.game_over = True
        
        return reward_a, reward_b, self.game_over
    
    def reset(self):
        """Reset game to initial state"""
        self.board = CarBoard(size=self.size)
        self.current_turn = 0
        self.game_over = False
        self.crashed = False
        self.score_a = 0
        self.score_b = 0
    
    def get_state(self):
        """Get current state for neural network"""
        return self.board.get_state_vector()
    
    def display(self):
        """Display the current board state"""
        return self.board.render()
    
    def get_info(self):
        """Get game information"""
        return {
            'turn': self.current_turn,
            'score_a': self.score_a,
            'score_b': self.score_b,
            'crashed': self.crashed,
            'game_over': self.game_over,
            'car_a_pos': self.board.car_a_pos,
            'car_b_pos': self.board.car_b_pos
        }
