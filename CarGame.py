from CarBoard import CarBoard, addTuple
import numpy as np

class CarGame:
    """
    Two-player car crash game
    - Car A (pursuer) tries to crash into Car B
    - Car B (evader) tries to avoid Car A
    - Both move simultaneously
    - Players earn points for squares they land on
    - Crash gives A +10, B -10
    """
    
    # Action mapping: 0=up, 1=down, 2=left, 3=right 
    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    
    def __init__(self, size=5, max_turns=50):
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
        Returns list of action indices (0=up, 1=down, 2=left, 3=right)
        """
        pos = self.board.car_a_pos if car == 'A' else self.board.car_b_pos
        valid = []
        
        for action_idx, delta in self.ACTIONS.items():
            new_pos = addTuple(pos, delta)
            if self.board._is_valid_pos(new_pos):
                valid.append(action_idx)
        
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
            reward_a += 10  # A gets bonus for crashing
            reward_b -= 10  # B gets penalty for being caught
        
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
