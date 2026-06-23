import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAi, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0  # controll randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # create queue 

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAi):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int) # convert to 0/1 values
    
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
        self.epsilon = 80 - self.num_games # more games -> smaller epsilon

        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            idx_move = random.randint(0, 2)
            final_move[idx_move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
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

            print(f"Gane: {agent.num_games}, score: {score}, record: {record}")
            
            plot_loss.append(loss)
            plot_scores.append(score)
            last_50_scores = plot_scores[-50:]
            plot_mean_last_50_scores.append(sum(last_50_scores) / len(last_50_scores))
            plot(plot_scores, plot_mean_last_50_scores, plot_loss)


if __name__ == '__main__':
    train()