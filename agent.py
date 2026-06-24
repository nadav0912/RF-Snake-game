import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAi, Direction, Point, get_idx, BLOCK_SIZE
from model import Conv_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 
BATCH_SIZE = 64
LR = 0.00025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0  # controll randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # create queue 

        self.model = Conv_QNet((3, 24+2, 32+2), 4).to(device)

        # Target model with whights of the model and in eval mode
        self.target_model = Conv_QNet((3, 26, 34), 4).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)


    def get_state(self, game: SnakeGameAi):
        cols = int(game.w // BLOCK_SIZE)
        rows = int(game.h // BLOCK_SIZE)

        # Create state with 4 channels: 0 -> head, 1 -> snake body, 2 -> apple, 3 -> direction
        state = np.zeros((3, rows+2, cols+2), dtype=int)

        # Set border of ones
        state[1, 0, :] = 1          # seiling
        state[1, -1, :] = 1         # floor
        state[1, :, 0] = 1          # left wall
        state[1, :, -1] = 1         # right wall


        # Set head in first channel
        head_x, head_y = get_idx(game.snake[0])
        if 0 <= head_x < cols and 0 <= head_y < rows:
            state[0, head_y + 1, head_x + 1] = 1

        # Set snake body channel
        for point in game.snake[1:]:
            bx, by = get_idx(point)
            state[1, by + 1, bx + 1] = 1

        # Set food channel
        food_x, food_y = get_idx(game.food)
        if 0 <= food_x < cols and 0 <= food_y < rows:
            state[2, food_y + 1, food_x + 1] = 1

        """
        # Set all values in 4 channel by the snake current direction 
        dir_map = {Direction.UP: 1, Direction.RIGHT: 2, Direction.DOWN: 3, Direction.LEFT: 4}
        state[3, :, :] = dir_map[game.direction]
        """

        return state
    
    def remember(self, state, action, reward, next_state, done):
        # Pop left if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        # Random moves - tradeoff exploration / exploition
        self.epsilon = 0 if self.num_games > 2000 else max(20, 1000 - self.num_games) # more games -> smaller epsilon
        
        """
        final_move = [0, 0, 0]
        if random.randint(0, 300) < self.epsilon:
            idx_move = random.randint(0, 2)
            final_move[idx_move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            state0 = torch.unsqueeze(state0, 0).to(device)
            prediction = self.model(state0)
            idx_move = torch.argmax(prediction).item()
            final_move[idx_move] = 1
        """

        final_move = [0, 0, 0, 0]
        if random.randint(0, 1200) < self.epsilon:
            idx_move = random.randint(0, 3)
            final_move[idx_move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            state0 = torch.unsqueeze(state0, 0).to(device)
            prediction = self.model(state0)
            idx_move = torch.argmax(prediction).item()
            final_move[idx_move] = 1

        return final_move
        

def train():
    plot_scores = []
    plot_mean_last_50_scores = []
    plot_loss = []
    record = 0
    agent = Agent()
    game = SnakeGameAi()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # store in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        # train long memory, plot result
        if done:
            game.reset()
            agent.num_games += 1
            loss = agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # Target Network Synchronization every 10 games
            if agent.num_games % 10 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

            print(f"Game: {agent.num_games}, score: {score}, record: {record}")
            
            plot_loss.append(loss)
            plot_scores.append(score)
            last_50_scores = plot_scores[-50:]
            plot_mean_last_50_scores.append(sum(last_50_scores) / len(last_50_scores))
            plot(plot_scores, plot_mean_last_50_scores, plot_loss)


if __name__ == '__main__':
    train()