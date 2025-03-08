import PPO
import gym
import argparse
import numpy as np
from itertools import count


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    env = gym.make('CartPole-v1', render_mode='human' if not args.train else None)
    agent = PPO.Agent(hyperparameter_set=args.hyperparameters)

    avg_score = 0
    steps = 0
    l_iterations = 0
    score_history = []
    best_score = env.reward_range[0]

    if args.train:
        for episode in count():
            state,_ = env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(state)
                new_state, reward, done, _, info = env.step(action)
                steps += 1
                score += reward
                agent.remember(state, action, prob, val, reward, done)
                if steps % 20 == 0:
                    agent.learn()
                    l_iterations += 1
                state = new_state
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score)
    else:
        agent.load_models()
        agent.critic.eval()
        agent.actor.eval()

        for episode in count():
            state,_ = env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(state)
                new_state, reward, done, _, info = env.step(action)
                score += reward
                state = new_state
            print('episode', episode, 'score %.1f' % score)
