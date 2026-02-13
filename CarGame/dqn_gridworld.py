"""
Deep Q-learning for CarGame - Complete Implementation
Includes Vanilla Model, Experience Replay, and Target Network
"""

import numpy as np
import torch
from Gridworld import Gridworld
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque
import copy


# ============================================================================
# VANILLA DQN MODEL
# ============================================================================

def train_vanilla_model():
    """Train a basic DQN model without experience replay"""
    print("Training Vanilla DQN Model...")
    
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 1.0

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    epochs = 100
    losses = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state1 = torch.from_numpy(state_).float()
        status = 1
        
        while(status == 1):
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            
            with torch.no_grad():
                newQ = model(state2.reshape(1, 64))
            maxQ = torch.max(newQ)
            
            if reward == -1:
                Y = reward + (gamma * maxQ)
            else:
                Y = reward
                
            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action_]
            loss = loss_fn(X, Y)
            print(f"Epoch {i}, Loss: {loss.item()}")
            clear_output(wait=True)
            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            
            if reward != -1:
                status = 0
                
        if epsilon > 0.1:
            epsilon -= (1 / epochs)
    
    # Plot losses
    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.title("Vanilla DQN Training Loss")
    plt.show()
    
    return model


# ============================================================================
# EXPERIENCE REPLAY DQN MODEL
# ============================================================================

def train_with_experience_replay():
    """Train DQN with experience replay to eliminate catastrophic forgetting"""
    print("Training DQN with Experience Replay...")
    
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    epochs = 5000
    losses = []
    mem_size = 1000  # Total size of experience replay memory
    batch_size = 200  # Minibatch size
    replay = deque(maxlen=mem_size)  # Create memory replay as deque list
    max_moves = 50  # Maximum number of moves before game is over
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            mov += 1
            qval = model(state1)  # Compute Q-values from input state
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):  # Epsilon-greedy strategy
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1, action_, reward, state2, done)  # Create experience tuple
            replay.append(exp)  # Add experience to replay memory
            state1 = state2
            
            if len(replay) > batch_size:  # Begin minibatch training
                minibatch = random.sample(replay, batch_size)  # Randomly sample
                
                # Separate components into minibatch tensors
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = model(state1_batch)  # Re-compute Q-values for states
                with torch.no_grad():
                    Q2 = model(state2_batch)  # Compute Q-values for next states
                
                # Compute target Q-values
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                print(f"Epoch {i}, Loss: {loss.item()}")
                clear_output(wait=True)
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if reward != -1 or mov > max_moves:  # If game is over
                status = 0
                mov = 0
                
    losses = np.array(losses)
    
    # Plot losses
    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.title("Experience Replay DQN Training Loss")
    plt.show()
    
    return model


# ============================================================================
# TARGET NETWORK DQN MODEL
# ============================================================================

def train_with_target_network():
    """Train DQN with target network to handle learning instability"""
    print("Training DQN with Target Network...")
    
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )

    model2 = copy.deepcopy(model)  # Create target network
    model2.load_state_dict(model.state_dict())  # Copy parameters

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    epochs = 5000
    losses = []
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    sync_freq = 500  # Update frequency for synchronizing target model
    j = 0
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            j += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)  # Use target network
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                print(f"Epoch {i}, Loss: {loss.item()}")
                clear_output(wait=True)
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if j % sync_freq == 0:  # Copy main model to target network
                    model2.load_state_dict(model.state_dict())
                    
            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0
                
    losses = np.array(losses)
    
    # Plot losses
    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.title("Target Network DQN Training Loss")
    plt.show()
    
    return model


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_model(model, mode='static', display=True):
    """Test the trained model"""
    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }
    
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()
    
    if display:
        print("Initial State:")
        print(test_game.display())
        
    status = 1
    while(status == 1):
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
            
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        
        if display:
            print(test_game.display())
            
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break
    
    win = True if status == 2 else False
    return win


def evaluate_model(model, mode='random', max_games=1000):
    """Evaluate model performance over multiple games"""
    wins = 0
    for i in range(max_games):
        win = test_model(model, mode=mode, display=False)
        if win:
            wins += 1
    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games, wins))
    print("Win percentage: {}%".format(100.0 * win_perc))
    return win_perc


def running_mean(x, N=50):
    """Calculate running mean for smoothing loss plots"""
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv) / N
    return y


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Choose which model to train:
    # 1 = Vanilla DQN
    # 2 = Experience Replay DQN
    # 3 = Target Network DQN
    
    model_choice = 3  # Change this to select different models
    
    if model_choice == 1:
        model = train_vanilla_model()
        print("\nTesting Vanilla Model:")
        test_model(model, mode='static')
        
    elif model_choice == 2:
        model = train_with_experience_replay()
        print("\nTesting Experience Replay Model:")
        test_model(model, mode='random')
        evaluate_model(model, mode='random', max_games=1000)
        
    elif model_choice == 3:
        model = train_with_target_network()
        print("\nTesting Target Network Model:")
        test_model(model, mode='random')
        evaluate_model(model, mode='random', max_games=1000)
