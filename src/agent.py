import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAi, Direction, Point, get_idx, BLOCK_SIZE
from model import Conv_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 
BATCH_SIZE = 256
LR = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, start_over=True):
        self.num_games = 0
        self.n_steps = 0
        self.epsilon = 0  # controll randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # create queue 

        self.model = Conv_QNet((3, 24+2, 32+2), 12, 4).to(device)

        if not start_over:
            self.model.load()
            self.num_games = 1200

        # Target model with whights of the model and in eval mode
        self.target_model = Conv_QNet((3, 26, 34), 12, 4).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    def get_image_state(self, game: SnakeGameAi):
        cols = int(game.w // BLOCK_SIZE)
        rows = int(game.h // BLOCK_SIZE)

        # Create state with 3 channels: 0 -> head, 1 -> snake body and walls, 2 -> apple
        state = np.zeros((3, rows+2, cols+2), dtype=np.float32)

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
            apple_y, apple_x = food_y + 1, food_x + 1
            
            # Create heat map with the apple as the center 
            # close to the center -> value close to 1, far from the center value decrese and get close to 0
            y_grid, x_grid = np.ogrid[:rows+2, :cols+2]
            dist_matrix = np.abs(y_grid - apple_y) + np.abs(x_grid - apple_x)
            max_dist = float(rows + cols)
            heatmap = np.maximum(0.0, 1.0 - (dist_matrix / max_dist))
            state[2, :, :] = heatmap
        else:
            raise ValueError("Apple not in screen border")

        return state
    
    def get_logic_state(self, game: SnakeGameAi):
        head = game.head
        
        # Calc 4 point around the snake head
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        
        state = [
            # Danger around the head
            game.is_collision(point_u),
            game.is_collision(point_r),
            game.is_collision(point_d),
            game.is_collision(point_l),
            
            # Current snake direction
            game.direction == Direction.UP,
            game.direction == Direction.RIGHT,
            game.direction == Direction.DOWN,
            game.direction == Direction.LEFT,
            
            # The position of the apple in relation to the snake head
            game.food.y < game.head.y,  # up
            game.food.x > game.head.x,  # right
            game.food.y > game.head.y,  # down
            game.food.x < game.head.x   # left
        ]
        
        return np.array(state, dtype=np.float32)

    def remember(self, state_img, state_logic, action, reward, next_state_img, next_state_logic, done):
        # Pop left if MAX_MEMORY is reached
        self.memory.append((state_img, state_logic, action, reward, next_state_img, next_state_logic, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory

        states_img, states_logic, actions, rewards, next_states_img, next_states_logic, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states_img, states_logic, actions, rewards, next_states_img, next_states_logic, dones)
        return loss

    def train_short_memory(self, state_img, state_logic, action, reward, next_state_img, next_state_logic, done):
        self.trainer.train_step(state_img, state_logic, action, reward, next_state_img, next_state_logic, done)
   
    def get_action(self, state_img, state_logic, game: SnakeGameAi):
        # Random moves - tradeoff exploration / exploition
        if self.num_games > 5000:
            self.epsilon = 0.0
        else:
            self.epsilon = max(0.02, 1.0 - (self.num_games / 4000))        
       
        final_move = [0, 0, 0, 0]
        if random.random() < self.epsilon:
            # Remove the opsite direction move to prevent him make a invalid move
            possible_moves = [0, 1, 2, 3]
            current_direction = game.direction
            if current_direction == Direction.UP: possible_moves.remove(2)
            elif current_direction == Direction.RIGHT: possible_moves.remove(3)
            elif current_direction == Direction.DOWN: possible_moves.remove(0)
            elif current_direction == Direction.LEFT: possible_moves.remove(1)

            idx_move = random.choice(possible_moves)
            final_move[idx_move] = 1
        else:
            state0_img = torch.tensor(np.array(state_img), dtype=torch.float).unsqueeze(0).to(device)
            state0_logic = torch.tensor(np.array(state_logic), dtype=torch.float).unsqueeze(0).to(device)
            
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state0_img, state0_logic)
            self.model.train()

            current_direction = game.direction
            if current_direction == Direction.UP: 
                prediction[0][2] = -float('inf')  # Block down
            elif current_direction == Direction.RIGHT: 
                prediction[0][3] = -float('inf')  # Block left
            elif current_direction == Direction.DOWN: 
                prediction[0][0] = -float('inf')  # Block up
            elif current_direction == Direction.LEFT: 
                prediction[0][1] = -float('inf')  # Block right
            
            idx_move = torch.argmax(prediction).item()
            final_move[idx_move] = 1

        return final_move
        

def train():
    plot_scores = []
    plot_mean_last_50_scores = []
    plot_loss = []
    record = 0
    agent = Agent(start_over=True)
    game = SnakeGameAi()

    while True:
        # get old state
        state_old_img = agent.get_image_state(game)
        state_old_logic = agent.get_logic_state(game)

        # get move
        final_move = agent.get_action(state_old_img, state_old_logic, game)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        agent.n_steps += 1

        state_new_img = agent.get_image_state(game)
        state_new_logic = agent.get_logic_state(game)

        # train short memory
        agent.train_short_memory(state_old_img, state_old_logic, final_move, reward, state_new_img, state_new_logic, done)

        # store in memory
        agent.remember(state_old_img, state_old_logic, final_move, reward, state_new_img, state_new_logic, done)

        # Target Network Synchronization every 1000 snake steps
        if agent.n_steps % 1000 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        # train long memory, plot result
        if done:
            game.reset()
            agent.num_games += 1
            loss = agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.num_games}, score: {score}, record: {record}")
            
            plot_loss.append(loss)
            plot_scores.append(score)
            last_50_scores = plot_scores[-50:]
            plot_mean_last_50_scores.append(sum(last_50_scores) / len(last_50_scores))
            plot(plot_scores, plot_mean_last_50_scores, plot_loss)


if __name__ == '__main__':
    train()