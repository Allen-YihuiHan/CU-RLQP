from typing import List, Callable, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
np.random.seed(42)

'''
For REINFORNCE Monte-Carlo Policy-Gradient Control (episodic) in LN17.
'''


def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)


def oracle_value(p: float) -> float:
    """ True value of the first state
    Parameters:
        p: probability of the action 'right'.
    Returns:
        True value of the first state.

    """
    
    v = (p ** 2 - 3 * p + 4) / (p * (p - 1))

    return v


def oracle_plot():
    fig, ax = plt.subplots(1, 1)

    # Plot a graph
    p = np.linspace(0.01, 0.99, 100)
    y = oracle_value(p)
    ax.plot(p, y, color='red')

    # Find a maximum point, can also be done analytically by taking a derivative
    imax = np.argmax(y)
    pmax = p[imax]
    ymax = y[imax]
    ax.plot(pmax, ymax, color='green', marker="*", label="optimal point: f({0:.2f}) = {1:.2f}".format(pmax, ymax))
    ax.set_ylabel("Value of the first state")
    ax.set_xlabel("Probability of the action 'right'")
    ax.set_title("Short corridor with switched actions")
    ax.set_ylim(ymin=-105.0, ymax=5)
    ax.legend()
    plt.savefig('img/Q2_oracle_value.png')


class ShortCorridor:
    """
    Short corridor environment
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0

    def step(self, action: int) -> Tuple[int, bool]:
        """
        Parameters:
            action: chosen action, 0 for left, 1 for right
        Returns:
            tuple of (reward, done)
        """
        
        assert self.state != 3, 'Agent already in terminal state'
        if self.state == 0 or self.state == 2:
            if action:
                self.state += 1
            else:
                self.state = max(0, self.state - 1)
        else:
            if action:
                self.state -= 1
            else:
                self.state += 1

        if self.state == 3:
            # terminal state
            return 0, True
        else:
            return -1, False


class Reinforce:
    """
    Reinforce that follows algorith 'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)' in LN17.
    Note that in this problem, all the states appear identical under the function approximation.
    To simplify your implementation, here we assume that h(a)=w^Ta, i.e., linear preference function.
    Thus, h(s,a)=h(a), x(s,a)=x(a), pi(s,a)=pi(a).
    """

    def __init__(self, alpha: float, gamma: float):
        # Set values such that initial conditions correspond to left-epsilon greedy
        self.theta = np.array([-1.47, 1.47])
        
        # Learning rate
        self.alpha = alpha
        
        # Discount Factor
        self.gamma = gamma
        
        # X(s,a), Feature State-Action vector for all states, first column - left, second - right
        self.x = np.array([[0, 1],
                           [1, 0]])

        self.rewards = []
        self.actions = []

    def get_pi(self) -> np.array:
        '''
        Returns:
            pi, 1-D
        '''
        
        h = self.x @ self.theta

        pi = np.exp(h) / np.exp(h).sum()

        # Ensure that the policy is stochastic for exploration
        imin = np.argmin(pi)
        epsilon = 0.05
        if pi[imin] < epsilon:
            pi[:] = 1 - epsilon
            pi[imin] = epsilon

        return pi

    def store_reward(self, reward):
        if reward is not None:
            self.rewards.append(reward)

    def store_action(self, action):
        if action is not None:
            self.actions.append(action)

    def choose_action(self):
        '''
        Returns:
            action, int in {0 , 1} or Boolean 
        '''
        
        pi = self.get_pi()

        choice = np.random.uniform(0,1)
        action = 0 if choice < pi[0] else 1

        return action

    def compute_discounted_return(self) -> np.array:
        '''
        Returns:
            G, 1-D
        '''
        
        G = np.zeros(len(self.rewards))
        for i in range(len(self.rewards)):
            discounted_return = [self.rewards[j] * self.gamma ** (j - i) for j in range(i, len(self.rewards))]
            G[i] = sum(discounted_return)

        return G

    def learn(self):
        # Update policy

        # Compute discounted return
        G = self.compute_discounted_return()

        # Update theta
        for i in range(len(G)):
            gradient = self.x[self.actions[i]] - np.exp(self.theta) / sum(np.exp(self.theta))
            self.theta += self.alpha * self.gamma ** i * G[i]*gradient

        self.rewards = []
        self.actions = []


def run(num_episodes: int, agent_generator: Callable[[], Reinforce]) -> Tuple[np.array, np.array]:
    # Train
    env = ShortCorridor()
    agent = agent_generator()

    # Define lists or dictionary for plotting metrics
    rewards = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        rewards_sum = 0
        reward = None
        env.reset()

        while True:
            # Take action
            action = agent.choose_action()
            reward, done = env.step(action)
            rewards_sum += reward

            # Store trajectory
            agent.store_reward(reward)
            agent.store_action(action)

            if done:
                # Update policy
                agent.learn()
                break

        rewards[episode_idx] = rewards_sum

    return rewards, agent.get_pi()


def REINFORCE_plot():
    num_trials = 100
    num_episodes = 1000
    gamma = 1
    agent_generators = [lambda: Reinforce(alpha=2e-4, gamma=gamma),
                        lambda: Reinforce(alpha=2e-5, gamma=gamma),
                        lambda: Reinforce(alpha=2e-3, gamma=gamma)]
    labels = ['alpha = 2e-4',
              'alpha = 2e-5',
              'alpha = 2e-3']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))
    policies = np.zeros((len(agent_generators), num_trials, 2))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward, policy = run(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward
            policies[agent_index, i, :] = policy

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    template = 'Final Optimal Policy for {}: Left {}, Right {}'
    
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
        print(template.format(label, policies[i].mean(axis=0)[0], policies[i].mean(axis=0)[1]))
        
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('img/Q2_reinforce.png')
    plt.close()

    n_lable = len(labels)
    left_p = [policies[i, :, 0].mean(axis=0) for i in range(n_lable)]
    right_p = [policies[i, :, 1].mean(axis=0) for i in range(n_lable)]
    plt.bar(range(n_lable), left_p, label='Left', fc='y')
    plt.bar(range(n_lable), right_p, bottom=left_p, label='Right', tick_label=labels, fc='b')
    plt.legend(loc='lower right')
    plt.savefig('img/Q2_reinforce_policy.png')
    plt.close()

if __name__ == '__main__':
    oracle_plot()
    REINFORCE_plot()
