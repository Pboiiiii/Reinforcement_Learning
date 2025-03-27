from DDPG import DDPG
from itertools import count
import gymnasium as gym
import numpy as np
import argparse


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    agent = DDPG(hyperparameter_set=args.hyperparameters)

    np.random.seed(0)

    score_history = []
    steps = 0

    if args.train:
        for i in count():
            done = False
            score = 0
            state, _ = env.reset()
            while not done:
                action = agent.choose_action(state)

                new_state, reward, done, _, info = env.step(action)

                agent.remember(state, action, reward, new_state, int(done))
                agent.learn()

                score += reward
                state = new_state
                steps += 1

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
            state, _ = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(state)
                new_state, reward, done, _, info = env.step(action)
                score += reward
                state = new_state
            print('episode', episode, 'score %.1f' % score)
