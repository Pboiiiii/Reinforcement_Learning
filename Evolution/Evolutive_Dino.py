from DQN import DQN
from itertools import count
from datetime import datetime
import random
import pygame
import os
import numpy as np
import torch
import argparse


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "Evolutive_Agents"
os.makedirs(RUNS_DIR, exist_ok=True)


pygame.init()

# Global variables
game_speed = 20
obstacles = []
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
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

        # Ducking action
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
    def __init__(self, image, type):
        self.image = image
        self.type = type
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
        screen.blit(self.image[self.index//5], self.rect)
        self.index += 1


class EVOLUTION:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.action_space = 3
        self.state_space = 5
        self.clock = pygame.time.Clock()
        self.cloud = Cloud()
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.death = False
        self.alternate = False
        self.generation = 0
        self.best_score = 0

        # Create the population
        self.population = [DQN(self.state_space, self.action_space, 10) for _ in range(num_agents)]

        # Create the dinosaurs
        self.dinos = [Dinosaur() for _ in range(num_agents)]

        # Lists to store the scores and deaths of each agent
        self.scores = [0]*num_agents
        self.deaths = [False]*num_agents

        # Path to save logs
        self.LOG_FILE = os.path.join(RUNS_DIR, f'Evolution.log')

        # Path to save all models
        self.MODELS_FILE = os.path.join(RUNS_DIR, f'Agents.pt')

    def score(self, gen):
        global game_speed
        self.points += 1
        if self.points % 100 == 0:
            game_speed += 1

        text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

        gen_text = self.font.render("Gen: " + str(gen), True, (0, 0, 0))
        gen_text_rect = gen_text.get_rect()
        gen_text_rect.center = (1000, 80)
        SCREEN.blit(gen_text, gen_text_rect)


    def background(self):
        global game_speed
        image_width = BG.get_width()
        SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0
        self.x_pos_bg -= game_speed


    @staticmethod
    def reset():
        """
        Returns initial observation space:
        (game speed, default player x pos, default player y pos, obstacle x pos, obstacle y pos)
        """
        global game_speed

        obs = [game_speed, 80, 310, 1100, 325]
        obs = np.array(obs)
        return obs


    def step(self, agent, action, index):
        """
        - Performs action on the current agent
        - Kills the agent if it has collided with an obstacle
        - Returns observation space:
            (game speed, player x pos, player y pos, obstacle x pos, obstacle y pos)
        """
        global game_speed

        agent.update(action)

        obs = [game_speed,
               agent.dino_rect.x,
               agent.dino_rect.y,
               # If there are no obstacles, set the obstacle x position to the rightmost side of the screen
               obstacles[0].rect.x if len(obstacles) > 0 else 1100,
               # If there are no obstacles, set the obstacle y position to the default level
               obstacles[0].rect.y if len(obstacles) > 0 else 325]

        obs = np.array(obs)

        if self.death:
            self.scores[index] = self.points
            self.deaths[index] = True

        return obs


    def Fitness_Selection(self):
        """
        - Performs fitness selection:
            Sorts the population based on their scores (from highest to lowest)
        """
        indexed_scores = sorted(enumerate(self.scores), key=lambda x: x[1], reverse=True)
        best_fit_order = [i[0] for i in indexed_scores]

        return best_fit_order


    @staticmethod
    def mutate(agent, mutation_rate=0.2):
        for param in agent.parameters():
            flattened_weights = param.data.view(-1)

            for i in range(flattened_weights.numel()):  # Use numel() for correct size
                if np.random.rand() < mutation_rate:
                    binary = agent.float_to_binary(flattened_weights[i].item())
                    bit_idx = np.random.randint(0, 32)
                    mutated_binary = (
                            binary[:bit_idx] +
                            ('0' if binary[bit_idx] == '1' else '1') +
                            binary[bit_idx + 1:]
                    )
                    mutated_value = agent.binary_to_float(mutated_binary)
                    flattened_weights[i] = torch.tensor(mutated_value, dtype=torch.float32)

            param.data = flattened_weights.view(param.shape)  # Ensure correct reshaping


    def crossover(self, parent1, parent2, crossover_rate=0.5):

        offspring1 = DQN(self.state_space, self.action_space, 10)
        offspring2 = DQN(self.state_space, self.action_space, 10)

        for (p1, p2, po1, po2) in zip(parent1.parameters(), parent2.parameters(), offspring1.parameters(),
                                      offspring2.parameters()):
            p1_flat = p1.data.view(-1).tolist()
            p2_flat = p2.data.view(-1).tolist()

            new_values_1, new_values_2 = [], []
            for w1, w2 in zip(p1_flat, p2_flat):
                if np.random.rand() < crossover_rate:  # Apply crossover with probability
                    # Convert to binary
                    binary1 = parent1.float_to_binary(w1)
                    binary2 = parent2.float_to_binary(w2)

                    # Choose a random crossover point
                    crossover_point = np.random.randint(1, 31)  # Avoid full swaps
                    offspring_binary_1 = binary1[:crossover_point] + binary2[crossover_point:]
                    offspring_binary_2 = binary2[:crossover_point] + binary1[crossover_point:]

                    # Convert back to float
                    new_value_1 = parent1.binary_to_float(offspring_binary_1)
                    new_value_2 = parent2.binary_to_float(offspring_binary_2)
                else:
                    new_value_1, new_value_2 = w1, w2  # No crossover, keep original values

                new_values_1.append(new_value_1)
                new_values_2.append(new_value_2)

            # Assign the new values to the offspring
            po1.data = torch.tensor(new_values_1[:po1.numel()], dtype=torch.float32).view(po1.shape)
            po2.data = torch.tensor(new_values_2[:po2.numel()], dtype=torch.float32).view(po2.shape)

        return offspring1, offspring2


    def running(self, training=True):
        """
         - Performs training or testing
        """
        global game_speed, obstacles

        start_time = datetime.now()

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')

        if training:
            # Checks whether a model file exists (if one exist, all models exist)
            if os.path.exists(os.path.join(f'D:\Python weas\RL\DQN_w_pytorch\{RUNS_DIR}', f'Agent1.pt')):
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Models file found, "
                               f"training from last generation...")

                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                # Loading models, last generation and best score
                checkpoint = torch.load(self.MODELS_FILE)
                for agent in range(self.num_agents):
                    self.population[agent].load_state_dict(checkpoint[f'model_state_dict{agent}'])
                    self.scores[agent] = checkpoint['score'][agent]
                self.best_score = max(self.scores)
                self.generation = checkpoint['generation']
            else:
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Models file not found, "
                               f"training from scratch...")
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

            for gen in count():
                self.deaths = [False] * self.num_agents
                self.clock = pygame.time.Clock()
                self.points = 0
                self.x_pos_bg = 0
                self.y_pos_bg = 380
                game_speed = 20
                obstacles = []

                # Initialize states for all agents
                self.states = [torch.tensor(self.reset(), dtype=torch.float32)] * self.num_agents

                # Loop until all agents are dead
                while not all(self.deaths):
                    SCREEN.fill((255, 255, 255))

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

                        # Iterate over all agents
                        for agent in range(self.num_agents):
                            self.death = False

                            # Check if current agent is dead
                            if not self.deaths[agent]:
                                self.dinos[agent].draw(SCREEN)

                                if self.dinos[agent].dino_rect.colliderect(obstacle.rect):
                                    self.death = True

                                # Selecting best action
                                with torch.no_grad():
                                    action = self.population[agent](self.states[agent].unsqueeze(dim=0)).squeeze().argmax()

                                # Excecuting action
                                new_state = self.step(self.dinos[agent], action.item(), agent)
                                new_state = torch.tensor(new_state, dtype=torch.float32)

                                # Moving to next state
                                self.states[agent] = new_state


                    self.background()

                    self.cloud.draw(SCREEN)
                    self.cloud.update()

                    self.score(gen + 1 + self.generation)

                    self.clock.tick(30)
                    pygame.display.update()

# __________________________ END OF GENERATION __________________________

                # Best score set in the generation
                gen_best_score = max(self.scores)

                # Resetting all dinos
                for agent in range(self.num_agents):
                    self.dinos[agent].dino_run = True
                    self.dinos[agent].dino_duck = False
                    self.dinos[agent].dino_jump = False
                    self.dinos[agent].jump_vel = self.dinos[agent].JUMP_VEL

                # Selecting the best agents based on fitness selection
                best_agents= self.Fitness_Selection()

                # Perform crossover if a higher score is reached
                if gen_best_score > self.best_score:

                    # Assign weights based on position (closer to left has higher weight)
                    weights = [1 / (i + 1.4) for i in range(len(best_agents))]

                    # Future new population (two best agents are kept)
                    childs = [self.population[best_agents[0]], self.population[best_agents[1]]]

                    while len(childs) < self.num_agents:

                        # Selecting parents (they cannot be the same)
                        parent1_idx, parent2_idx = random.choices(best_agents, weights=weights, k=2)
                        while parent1_idx == parent2_idx:
                            _, parent2_idx = random.choices(best_agents, weights=weights, k=2)

                        # Crossover is applied given the two parents
                        child1, child2 = self.crossover(self.population[parent1_idx],
                                                        self.population[parent2_idx])

                        childs.append(child1)
                        childs.append(child2)

                    # Replace the old population with the new population
                    self.population = childs[:]

                    # Update best score
                    self.best_score = gen_best_score

                    if self.alternate:
                        log_message = (f"{start_time.strftime(DATE_FORMAT)}: "
                                       f"New best score: {self.best_score}, performing crossover -"
                                       f" Generation: {gen + 1 + self.generation} - Saving models...")
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                        self.alternate = False

                        # Saving models
                        for agent in range(self.num_agents):
                            torch.save({f"model{agent}_state_dict": self.population[agent].state_dict(),
                                        "score": self.scores[agent]},
                                       self.MODELS_FILE)
                        torch.save({"generation": gen + 1 + self.generation}, self.MODELS_FILE)

                else:
                    # Performs mutation otherwise
                    for agent in range(self.num_agents):
                        self.mutate(self.population[agent])

                    if not self.alternate:
                        log_message = (f"{start_time.strftime(DATE_FORMAT)}: "
                                       f"Not much improvement, performing mutation "
                                       f"- Generation: {gen + 1 + self.generation} - Saving models...")
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                        self.alternate = True

                        for agent in range(self.num_agents):
                            torch.save({f"model{agent}_state_dict": self.population[agent].state_dict(),
                                        "score": self.scores[agent]},
                                       self.MODELS_FILE)
                        torch.save({"generation": gen + 1 + self.generation}, self.MODELS_FILE)


        else:
            # Loading models
            checkpoint = torch.load(self.MODELS_FILE)
            for agent in range(self.num_agents):
                self.population[agent].load_state_dict(checkpoint[f'model{agent}_state_dict'])
                self.scores[agent] = checkpoint['score'][agent]
            self.generation = checkpoint['generation']

            # Determining the best agent to test
            best_agent = np.argmax(self.scores)
            print(f"Testing the best agent with a score of: {self.scores[best_agent]}")

            # Switch best model to evaluation mode
            self.population[best_agent].eval()

            while True:
                SCREEN.fill((255, 255, 255))

                self.dinos[best_agent].draw(SCREEN)

                with torch.no_grad():
                    action = self.population[best_agent](self.states[best_agent].unsqueeze(dim=0)).squeeze().argmax()

                self.dinos[best_agent].update(action)

                for obstacle in obstacles:
                    obstacle.draw(SCREEN)
                    obstacle.update()
                    if self.dinos[best_agent].dino_rect.colliderect(obstacle.rect):
                        pygame.time.delay(500)
                        self.points = 0
                        self.x_pos_bg = 0
                        self.y_pos_bg = 380
                        game_speed = 20
                        self.points = 0
                        obstacles = []
                        self.dinos[best_agent].dino_run = True
                        self.dinos[best_agent].dino_jump = False
                        self.dinos[best_agent].dino_duck = False
                        continue

                self.background()

                self.cloud.draw(SCREEN)
                self.cloud.update()

                self.score(1 + self.generation)

                self.clock.tick(30)
                pygame.display.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    game = EVOLUTION(num_agents=50)

    if args.train:
        game.running(training=True)
    else:
        game.running(training=False)

