import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# DQN Network for Chess
class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# Helper function: Encode the board into a vector
def board_to_vector(board):
    board_vec = np.zeros(64, dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = {
                chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
                chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
            }[piece.piece_type]
            board_vec[i] = piece_value * (1 if piece.color == chess.WHITE else -1)
    return board_vec

def get_legal_moves(board):
    return list(board.legal_moves)

def train(episodes=10, batch_size=32, gamma=0.99):  # Reduce episodes for quick preview
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 64
    max_moves = 100
    buffer = ReplayBuffer(5000)
    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.05

    max_actions = 218  # Max possible legal moves in Chess
    dqn = DQN(input_size, max_actions).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)

    for episode in range(episodes):
        board = chess.Board()
        total_reward = 0
        done = False
        move_count = 0

        while not done and move_count < max_moves and not board.is_game_over():
            state_vec = board_to_vector(board)
            legal_moves = get_legal_moves(board)
            padded_moves = legal_moves + [None]*(max_actions - len(legal_moves))

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action_index = random.randint(0, len(legal_moves)-1)
                move = legal_moves[action_index]
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                    q_values = dqn(state_tensor)
                    mask = torch.zeros(max_actions).to(device)
                    mask[:len(legal_moves)] = 1
                    q_values_masked = q_values * mask
                    action_index = q_values_masked.argmax().item()
                    move = padded_moves[action_index]
                    if move is None:
                        action_index = random.randint(0, len(legal_moves)-1)
                        move = legal_moves[action_index]

            board.push(move)
            move_count += 1

            # Reward
            if board.is_game_over():
                result = board.result()
                if result == '1-0':
                    reward = 1 if board.turn == chess.WHITE else -1
                elif result == '0-1':
                    reward = -1 if board.turn == chess.WHITE else 1
                else:
                    reward = 0
                done = True
            else:
                reward = -0.01

            next_state_vec = board_to_vector(board)
            buffer.push(state_vec, action_index, reward, next_state_vec, done)
            total_reward += reward

            # Learn
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = dqn(states).gather(1, actions).squeeze()
                next_q_values = dqn(next_states).max(1)[0]
                expected_q = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f} | Final board FEN: {board.fen()}")

if __name__ == "__main__":
    print("Training Chess DQN Agent for preview ...")
    train()
    print("Preview complete. See above for episode rewards and final board positions.")
