import numpy as np

class CarBoard:
    """
    Board for the car crash game
    - Normalized coordinate system [0, 1] x [0, 1]
    - Two cars: A (pursuer) starts top-left, B (evader) starts bottom-right
    - Each square has points = 2^(Manhattan distance to nearest corner)
    - Movements are scaled by 1/size
    """
    
    def __init__(self, size=5):
        if size < 3 or size % 2 == 0:
            raise ValueError("Size must be odd and >= 3")
        
        self.size = size
        self.step_size = 1.0 / size  # Movement per action
        self.crash_threshold = self.step_size * 0.6  # Cars crash if within this distance
        
        # Normalized positions [0, 1]
        self.car_a_pos = (0.0, 0.0)  # Top-left
        self.car_b_pos = (1.0, 1.0)  # Bottom-right
        
        # Pre-calculate point values for each square
        self.point_grid = self._calculate_point_grid()
    
    def _calculate_point_grid(self):
        """Calculate points normalized to [0,1] based on Manhattan distance to nearest corner"""
        grid = np.zeros((self.size, self.size), dtype=float)
        corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
        
        # First calculate raw exponential values
        for i in range(self.size):
            for j in range(self.size):
                # Find minimum Manhattan distance to any corner
                min_dist = min(abs(i - ci) + abs(j - cj) for ci, cj in corners)
                grid[i, j] = 2 ** min_dist
        
        # Normalize to [0, 1] range
        min_val = grid.min()
        max_val = grid.max()
        if max_val > min_val:
            grid = (grid - min_val) / (max_val - min_val)
        
        return grid
    
    def _pos_to_grid(self, pos):
        """Convert normalized position [0,1] to discrete grid indices"""
        row = int(pos[0] * self.size)
        col = int(pos[1] * self.size)
        # Clamp to valid range
        row = max(0, min(row, self.size - 1))
        col = max(0, min(col, self.size - 1))
        return (row, col)
    
    def get_square_points(self, pos):
        """Get points for landing on a square (converts normalized pos to grid)"""
        grid_pos = self._pos_to_grid(pos)
        return self.point_grid[grid_pos]
    
    def move_car(self, car, new_pos):
        """Move a car to new position (clamped to [0,1] bounds)"""
        # Clamp to [0, 1] range
        clamped_pos = (max(0.0, min(1.0, new_pos[0])), max(0.0, min(1.0, new_pos[1])))
        
        if car == 'A':
            self.car_a_pos = clamped_pos
        elif car == 'B':
            self.car_b_pos = clamped_pos
        return True
    
    def _is_valid_pos(self, pos):
        """Check if position is within normalized bounds [0,1]"""
        return 0.0 <= pos[0] <= 1.0 and 0.0 <= pos[1] <= 1.0
    
    def is_crash(self):
        """Check if both cars are close enough to crash"""
        dist = np.sqrt((self.car_a_pos[0] - self.car_b_pos[0])**2 + 
                       (self.car_a_pos[1] - self.car_b_pos[1])**2)
        return dist < self.crash_threshold
    
    def render(self):
        """Return string representation of the board"""
        dtype = '<U2'
        board = np.zeros((self.size, self.size), dtype=dtype)
        board[:] = '·'
        
        # Convert positions to grid indices
        grid_a = self._pos_to_grid(self.car_a_pos)
        grid_b = self._pos_to_grid(self.car_b_pos)
        
        # Place cars
        if self.is_crash():
            board[grid_a] = 'X'  # Crash symbol
        else:
            board[grid_a] = 'A'
            if grid_a != grid_b:  # Only place B if not same grid square
                board[grid_b] = 'B'
        
        return board
    
    def render_np(self):
        """
        Return numpy representation for neural network
        Returns 2-channel grid: channel 0 = car A, channel 1 = car B
        """
        board = np.zeros((2, self.size, self.size), dtype=np.uint8)
        
        # Convert normalized positions to grid indices
        grid_a = self._pos_to_grid(self.car_a_pos)
        grid_b = self._pos_to_grid(self.car_b_pos)
        
        board[0, grid_a[0], grid_a[1]] = 1  # Car A position
        board[1, grid_b[0], grid_b[1]] = 1  # Car B position
        return board
    
    def get_state_vector(self):
        """Return flattened state for neural network input"""
        return self.render_np().flatten()


def addTuple(a, b):
    """Add two tuples element-wise"""
    return tuple([sum(x) for x in zip(a, b)])
