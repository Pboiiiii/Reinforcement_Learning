from Config import Config
from itertools import count
from DDPG import DDPG
import pybullet as p
import numpy as np
import argparse
import time
import os


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    model_path = "URDF_MODELS/test.urdf"

    env = Config(model_path, True)
    env.client_setup(iters=25)
    env.model_setup(damping=0.05)

    agent = DDPG(hyperparameter_set=args.hyperparameters)

    np.random.seed(0)

    score_history = []
    steps = 0


    if args.train:
        if os.path.exists("DDPG_robot_runs/DDPG_Actor.pt"):
            print("Model files found")
            agent.load_models()
        for i in count():
            steps += 1
            done = False
            score = 0
            state = env.reset()
            while not done:
                p.stepSimulation()
                time.sleep(1. / 240.)

                action = agent.choose_action(state)

                new_state, reward, done, info = env.step(action)

                agent.remember(state, action, reward, new_state, int(done))
                agent.learn()

                score += reward
                state = new_state

            score_history.append(score)
            print('episode', i, 'score %.1f' % score, '100 game average score %.2f'
                  % np.mean(score_history[-100:]))

            if steps % 25 == 0:
                agent.save_models()
    else:
        # Just in case
        agent.load_models()
        agent.eval()

        for episode in count():
            state = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(state)
                new_state, reward, done, info = env.step(action)
                score += reward
                state = new_state
            print('episode', episode, 'score %.1f' % score)
