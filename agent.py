import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import DQN, DoubleQTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01

class Agent:

    def __init__(self, name : str) -> None:
        self.n_games = 0
        self.gamma = 0.8 # discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(11, 3, name) 
        self.target_model = DQN(11, 3, name+'_target') 
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)
        self.trainer = DoubleQTrainer(model=self.model, target_model=self.target_model, lr=LR, gamma=self.gamma)
        
    def get_state(self, game : SnakeGameAI):
        head = game.snake[0]
        # used to sense danger in close proximity to the snake's head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or 
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)),

            # danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_d and game.is_collision(point_r)),

            # move directions 
            dir_l, 
            dir_r,
            dir_u, 
            dir_d,

            # food location
            game.food.x <= game.head.x, # food left
            game.food.x >= game.head.x, # food right
            game.food.y > game.head.y, # food down
            game.food.y < game.head.y, # food up
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if max memory reached

    def train_long_memory(self):
        mini_sample = self.memory if len(self.memory) <= BATCH_SIZE else random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
    
    def get_action(self, state):
        # random moves : tradeoff between exploration and exploitation
        epsilon = 0.5 * (0.99**self.n_games)
        if random.random() < epsilon:
            return random.randint(0, 2)
        state = torch.tensor(state, dtype=torch.float)
        return torch.argmax(self.model(state)).item()
    
    def save(self) -> None:
        self.model.save()
        self.target_model.save()