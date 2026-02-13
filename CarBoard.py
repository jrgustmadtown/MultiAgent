import numpy as np

class CarBoard:
    """
    Board for the car crash game
    - n x n grid (n is odd, n >= 3)
    - Two cars: A (pursuer) starts top-left, B (evader) starts bottom-right
    - Each square has points = 2^(Manhattan distance to nearest corner)
    """
    
    def __init__(self, size=5):
        if size < 3 or size % 2 == 0:
            raise ValueError("Size must be odd and >= 3")
        
        self.size = size
        self.car_a_pos = (0, 0)  # Top-left
        self.car_b_pos = (size - 1, size - 1)  # Bottom-right
        
        # Pre-calculate point values for each square
        self.point_grid = self._calculate_point_grid()
    
    def _calculate_point_grid(self):
        """Calculate 2^(Manhattan distance to nearest corner) for each square"""
        grid = np.zeros((self.size, self.size), dtype=int)
        corners = [
            (0, 0),
            (0, self.size - 1),
            (self.size - 1, 0),
            (self.size - 1, self.size - 1)
        ]
        
        for i in range(self.size):
            for j in range(self.size):
                # Find minimum Manhattan distance to any corner
                min_dist = min(abs(i - ci) + abs(j - cj) for ci, cj in corners)
                grid[i, j] = 2 ** min_dist
        
        return grid
    
    def get_square_points(self, pos):
        """Get points for landing on a square"""
        return self.point_grid[pos]
    
    def move_car(self, car, new_pos):
        """Move a car to new position (must be valid)"""
        if not self._is_valid_pos(new_pos):
            raise ValueError(f"Invalid position {new_pos} for car {car}")
        
        if car == 'A':
            self.car_a_pos = new_pos
        elif car == 'B':
            self.car_b_pos = new_pos
        return True
    
    def _is_valid_pos(self, pos):
        """Check if position is within grid bounds"""
        i, j = pos
        return 0 <= i < self.size and 0 <= j < self.size
    
    def is_crash(self):
        """Check if both cars occupy the same position"""
        return self.car_a_pos == self.car_b_pos
    
    def render(self):
        """Return string representation of the board"""
        dtype = '<U2'
        board = np.zeros((self.size, self.size), dtype=dtype)
        board[:] = '·'
        
        # Place cars
        if self.is_crash():
            board[self.car_a_pos] = 'X'  # Crash symbol
        else:
            board[self.car_a_pos] = 'A'
            board[self.car_b_pos] = 'B'
        
        return board
    
    def render_np(self):
        """
        Return numpy representation for neural network
        Returns 2-channel grid: channel 0 = car A, channel 1 = car B
        """
        board = np.zeros((2, self.size, self.size), dtype=np.uint8)
        board[0, self.car_a_pos[0], self.car_a_pos[1]] = 1  # Car A position
        board[1, self.car_b_pos[0], self.car_b_pos[1]] = 1  # Car B position
        return board
    
    def get_state_vector(self):
        """Return flattened state for neural network input"""
        return self.render_np().flatten()


def addTuple(a, b):
    """Add two tuples element-wise"""
    return tuple([sum(x) for x in zip(a, b)])
