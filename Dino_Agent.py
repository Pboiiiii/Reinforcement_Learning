import pygame
import os
import random
import numpy as np
import yaml
from DQN import DQN
from replay_mem import ReplayMemory
from itertools import count
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import argparse


# Global variables
game_speed = 20
obstacles = []
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

def INIT(render=True, agentname=None):
    global game_speed, obstacles, SCREEN_HEIGHT, SCREEN_WIDTH, DATE_FORMAT, RUNS_DIR, args

    # Prevent Pygame from creating a window
    if not render:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    pygame.init()

    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
               pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
    JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
    DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
               pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

    SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
    LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

    BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
            pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

    CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

    BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

    class Dinosaur:
        X_POS = 80
        Y_POS = 310
        Y_POS_DUCK = 340
        JUMP_VEL = 8.5

        def __init__(self):
            self.duck_img = DUCKING
            self.run_img = RUNNING
            self.jump_img = JUMPING

            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

            self.step_index = 0
            self.jump_vel = self.JUMP_VEL
            self.image = self.run_img[0]
            self.dino_rect = self.image.get_rect()
            self.dino_rect.x = self.X_POS
            self.dino_rect.y = self.Y_POS

        def update(self, action):
            if self.dino_duck:
                self.duck()
            if self.dino_run:
                self.run()
            if self.dino_jump:
                self.jump()

            if self.step_index >= 10:
                self.step_index = 0

            # Jumping action
            if action == 1 and not self.dino_jump:
                self.dino_duck = False
                self.dino_run = False
                self.dino_jump = True

            # Ducking action (atcs even in the air)
            elif action == 2:
                self.dino_duck = True
                self.dino_run = False
                self.dino_jump = False
                self.jump_vel = self.JUMP_VEL

            # Running action
            elif not (self.dino_jump or action == 2):
                self.dino_duck = False
                self.dino_run = True
                self.dino_jump = False

        def duck(self):
            self.image = self.duck_img[self.step_index // 5]
            self.dino_rect = self.image.get_rect()
            self.dino_rect.x = self.X_POS
            self.dino_rect.y = self.Y_POS_DUCK
            self.step_index += 1

        def run(self):
            self.image = self.run_img[self.step_index // 5]
            self.dino_rect = self.image.get_rect()
            self.dino_rect.x = self.X_POS
            self.dino_rect.y = self.Y_POS
            self.step_index += 1

        def jump(self):
            self.image = self.jump_img
            if self.dino_jump:
                self.dino_rect.y -= self.jump_vel * 4
                self.jump_vel -= 0.8
            if self.jump_vel < - self.JUMP_VEL:
                self.dino_jump = False
                self.jump_vel = self.JUMP_VEL

        def draw(self, screen):
            screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    class Cloud:
        def __init__(self):
            self.x = SCREEN_WIDTH + random.randint(800, 1000)
            self.y = random.randint(50, 100)
            self.image = CLOUD
            self.width = self.image.get_width()

        def update(self):
            self.x -= game_speed
            if self.x < -self.width:
                self.x = SCREEN_WIDTH + random.randint(2500, 3000)
                self.y = random.randint(50, 100)

        def draw(self, screen):
            screen.blit(self.image, (self.x, self.y))

    class Obstacle:
        def __init__(self, image, typ):
            self.image = image
            self.type = typ
            self.rect = self.image[self.type].get_rect()
            self.rect.x = SCREEN_WIDTH

        def update(self):
            self.rect.x -= game_speed
            if self.rect.x < -self.rect.width:
                obstacles.pop()

        def draw(self, screen):
            screen.blit(self.image[self.type], self.rect)

    class SmallCactus(Obstacle):
        def __init__(self, image):
            self.type = random.randint(0, 2)
            super().__init__(image, self.type)
            self.rect.y = 325

    class LargeCactus(Obstacle):
        def __init__(self, image):
            self.type = random.randint(0, 2)
            super().__init__(image, self.type)
            self.rect.y = 300

    class Bird(Obstacle):
        def __init__(self, image):
            self.type = 0
            super().__init__(image, self.type)
            self.rect.y = 250
            self.index = 0

        def draw(self, screen):
            if self.index >= 9:
                self.index = 0
            screen.blit(self.image[self.index // 5], self.rect)
            self.index += 1

    class game:
        def __init__(self, hyperparameter_set):
            # Reading hyperparameters
            with open('hyperparameters.yml', 'r') as f:
                all_hyperparameters_set = yaml.safe_load(f)
                self.hyperparameters = all_hyperparameters_set[hyperparameter_set]

            self.hyperparameter_set = hyperparameter_set

            # Hyperparameters (adjustable)
            # learning rate (alpha)
            self.learning_rate_a = self.hyperparameters['learning_rate_a']
            # discount rate (gamma)
            self.discount_factor_g = self.hyperparameters['discount_factor_g']
            # number of steps the agent takes before syncing the policy and target network
            self.network_sync_rate = self.hyperparameters['network_sync_rate']
            # size of replay memory
            self.replay_memory_size = self.hyperparameters['replay_memory_size']
            # size of the training data set sampled from the replay memory
            self.mini_batch_size = self.hyperparameters['mini_batch_size']
            # 1 = 100% random actions
            self.epsilon_init = self.hyperparameters['epsilon_init']
            # epsilon decay rate
            self.epsilon_decay = self.hyperparameters['epsilon_decay']
            # minimum epsilon value
            self.epsilon_min = self.hyperparameters['epsilon_min']
            # stop training after reaching this number of rewards
            self.stop_on_reward = self.hyperparameters['stop_on_reward']
            self.fc1_nodes = self.hyperparameters['fc1_nodes']

            # Neural Network
            self.loss_fn = nn.MSELoss()
            self.optimizer = None

            # Path to Run info
            self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
            self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
            self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

            # Game Variables
            self.run = True
            self.dino = Dinosaur()
            self.cloud = Cloud()
            self.clock = pygame.time.Clock()
            self.x_pos_bg = 0
            self.y_pos_bg = 380
            self.points = 0
            self.death_count = 0
            self.font = pygame.font.Font('freesansbold.ttf', 20)
            self.action = {0: self.dino, 1: 'duck', 2: 'run'}


        def score(self):
            global game_speed
            self.points += 1
            if self.points % 50 == 0:
                game_speed += 1

            text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)


        def background(self):
            image_width = BG.get_width()
            SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
            SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            if self.x_pos_bg <= -image_width:
                SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
                x_pos_bg = 0
            self.x_pos_bg -= game_speed


        # Performs an action and returns the observation, reward, and whether the game is over
        def step(self, action):
            global game_speed

            # Perform action
            self.dino.update(action)

            # Observation space: (game speed, player x pos, player y pos, obstacle x pos, obstacle y pos)
            obs = [game_speed,
                   self.dino.dino_rect.x,
                   self.dino.dino_rect.y,
                   # If there are no obstacles, set the obstacle x position to the rightmost side of the screen
                   obstacles[0].rect.x if len(obstacles) > 0 else 1100,
                   # If there are no obstacles, set the obstacle y position to the default level
                   obstacles[0].rect.y if len(obstacles) > 0 else 325]
            obs = np.array(obs)

            # Check if the game is over by death or by quitting
            terminated = False

            if self.death_count == 1:
                #print('Game Over')
                terminated = True
                self.points = self.points - 500

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            # Reward obtained
            reward = self.points

            return obs, reward, terminated


        # Returns initial observation values
        def reset(self):
            obs = [game_speed, self.dino.dino_rect.x, self.dino.dino_rect.y, 1100, 325]
            obs = np.array(obs)
            return obs


        # Optimize policy network
        def optimize(self, mini_batch, policy_dqn, target_dqn):

            # Transpose the list of experiences and separate each element
            states, actions, new_states, rewards, terminations = zip(*mini_batch)

            # Stack tensors to create batch tensors
            # tensor([[1,2,3]])
            states = torch.stack(states)

            actions = torch.stack(actions)

            new_states = torch.stack(new_states)

            rewards = torch.stack(rewards)
            terminations = torch.tensor(terminations).float()

            with torch.no_grad():
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

            # Calcuate Q values from current policy
            current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

            # Compute loss
            loss = self.loss_fn(current_q, target_q)

            # Optimize the model (backpropagation)
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update network parameters i.e. weights and biases


        # Graphs rewards per episode and epsilon decay
        def save_graph(self, rewards_per_episode, epsilon_history):
            # Save plots
            fig = plt.figure(1)

            # Plot average rewards (Y-axis) vs episodes (X-axis)
            mean_rewards = np.zeros(len(rewards_per_episode))
            for x in range(len(mean_rewards)):
                mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
            plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
            # plt.xlabel('Episodes')
            plt.ylabel('Mean Rewards')
            plt.plot(mean_rewards)

            # Plot epsilon decay (Y-axis) vs episodes (X-axis)
            plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
            # plt.xlabel('Time Steps')
            plt.ylabel('Epsilon Decay')
            plt.plot(epsilon_history)

            plt.subplots_adjust(wspace=1.0, hspace=1.0)

            # Save plots
            fig.savefig(self.GRAPH_FILE)
            plt.close(fig)


        # Runs the game until the game is over
        def running(self, is_training=True):
            global obstacles, game_speed
            if is_training:
                start_time = datetime.now()
                last_graph_update_time = start_time

                log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

            rewards_per_episode = []
            num_actions = 3
            num_states = 5

            policy_dqn = DQN(num_states, num_actions)

            if is_training:
                # Checks whether the model of the file already exist, if so, loads it and trains from then on
                if os.path.exists(os.path.join(f'D:\Python weas\RL\DQN_w_pytorch\{RUNS_DIR}', f'{self.hyperparameter_set}.pt')):
                    log_message = (f"Model file, {self.hyperparameter_set}.pt, found\n"
                                   f"Loading Model file: {self.hyperparameter_set}.pt")
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    # Loading the model
                    checkpoint = torch.load(self.MODEL_FILE)
                    policy_dqn.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epsilon = checkpoint['epsilon']
                    steps = checkpoint['step_count']
                    best_reward = checkpoint['best_reward']
                    memory = ReplayMemory(maxlen=self.replay_memory_size)
                    memory.load_state_dict(checkpoint['replay_memory'])
                else:
                    log_message = f"No model file found\nStarting from scratch"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    # Policy network optimizer, 'Adam' optimizer can be swapped with something else
                    self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

                    # Initialize epsilon
                    epsilon = self.epsilon_init

                    # Counting steps
                    steps = 0

                    # Track best reward
                    best_reward = -999999

                    # Initialize replay memory
                    memory = ReplayMemory(maxlen=self.replay_memory_size)

                # Create target network and make it identical to the policy network
                target_dqn = DQN(num_states, num_actions)
                target_dqn.load_state_dict(policy_dqn.state_dict())

                # List to keep track of the epsilon decay
                epsilon_history = []

            else:
                # Load learned policy
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

                # switch model to evaluation mode
                policy_dqn.eval()

            for episode in count():
                self.clock = pygame.time.Clock()
                self.dino = Dinosaur()
                self.cloud = Cloud()
                game_speed = 20
                self.x_pos_bg = 0
                self.y_pos_bg = 380
                self.points = 0
                obstacles = []
                self.death_count = 0

                self.run = True
                episode_reward = 0

                state = self.reset()
                state = torch.tensor(state, dtype=torch.float32)

                while self.run and episode_reward < self.stop_on_reward:
                    # Drawing and updating elements in the game
                    SCREEN.fill((255, 255, 255))
                    self.dino.draw(SCREEN)

                    if len(obstacles) == 0:
                        if random.randint(0, 2) == 0:
                            obstacles.append(SmallCactus(SMALL_CACTUS))
                        elif random.randint(0, 2) == 1:
                            obstacles.append(LargeCactus(LARGE_CACTUS))
                        elif random.randint(0, 2) == 2:
                            obstacles.append(Bird(BIRD))

                    for obstacle in obstacles:
                        obstacle.draw(SCREEN)
                        obstacle.update()
                        if self.dino.dino_rect.colliderect(obstacle.rect):
                            pygame.time.delay(1000)
                            self.death_count += 1

                    self.background()

                    self.cloud.draw(SCREEN)
                    self.cloud.update()

                    self.score()

                    self.clock.tick(30)
                    pygame.display.update()

# ___________________________________ Magic happens here ___________________________________

                    # Selecting action based on epsilon-greedy
                    if is_training and random.random() < epsilon:
                        # Selecting random action
                        action = random.randint(0, 2)
                        action = torch.tensor(action, dtype=torch.int64)
                    else:
                        # Selecting best action
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()


                    # Excecuting action
                    new_state, reward, terminated = self.step(action.item())

                    # Last best reward
                    episode_reward = reward

                    # Converting new state and reward to tensor
                    new_state = torch.tensor(new_state, dtype=torch.float32)
                    reward = torch.tensor(reward, dtype=torch.float32)

                    # Storing the experience in the replay memory
                    if is_training:
                        memory.append((state, action, new_state, reward, terminated))

                        # Increasing step counter
                        steps += 1

                    # Moving to new state
                    state = new_state

                    # Check if the game is over
                    if terminated:
                        self.run = False


                #Storing each reward accumulated in the episode
                rewards_per_episode.append(episode_reward)

                # Save model when new best reward is obtained.
                if is_training:
                    if episode_reward > best_reward:
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, saving model..."
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                        torch.save({
                            'model_state_dict': policy_dqn.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epsilon': epsilon,
                            'step_count': steps,
                            'best_reward': episode_reward,
                            'replay_memory': memory.state_dict()["memory"]
                        }, self.MODEL_FILE)

                        best_reward = episode_reward

                    # Update graph every x seconds
                    current_time = datetime.now()
                    if current_time - last_graph_update_time > timedelta(seconds=10):
                        self.save_graph(rewards_per_episode, epsilon_history)
                        last_graph_update_time = current_time

                    # If enough experience has been collected
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # Decay epsilon
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        # Copy policy network to target network after a certain number of steps
                        if steps > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0

            pygame.quit()

    # Running the game
    game = game(hyperparameter_set=agentname)

    if args.train:
        game.running(is_training=True)
    else:
        game.running(is_training=False)


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    if args.train:
        INIT(render=True, agentname=args.hyperparameters)
    else:
        INIT(render=True, agentname=args.hyperparameters)
