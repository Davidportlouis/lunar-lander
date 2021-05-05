import gym
from dqn import Agent
from utils import plot_learning_curve
import numpy as np

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
            eps_end=0.01, ip_dims=[8], lr=0.003)
    scores, eps = [], []
    n_games = 50

    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()
        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        scores.append(score)
        eps.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print(f"episode: {i} score: {score:.2f} avg_score: {avg_score:.2f} epsilon: {agent.epsilon:.2f}")
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x,scores,eps,"./assets/lunar_lander.png")
