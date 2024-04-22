import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class Actor(nn.Module):
    '''
    A simple Actor Network for discrete action space.
    '''

    def __init__(self, dim_state: int, n_action: int, n_hidden:int):
        """
        Parameter:
        ----------
        dim_state: dimension of x(s).
        n_action: number of possible actions at s.
        n_hidden: number of hidden neurons.
        """

        super(Actor, self).__init__()
        self.fc1 = nn.Linear(dim_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_action)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.tensor):
        """
        Parameter:
        ----------
        x: x(s), State Feature Vector.

        Returns:
        ----------
        prob: action selection probability.
        """
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        prob = self.softmax(x)
        return prob


class Critic(nn.Module):
    '''
    A simple Critic Network
    '''

    def __init__(self, dim_state, n_hidden):
        """
        Parameter:
        ----------
        n_action: int, number of possible actions at s.
        n_hidden: int, number of hidden neurons.
        """

        super(Critic, self).__init__()
        self.fc1 = nn.Linear(dim_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        """
        Parameter:
        ----------
        x: tensor, x(s), State Feature Vector.

        Returns:
        ----------
        v: tensor, v(s), state value.

        """
        
        x = self.fc1(x)
        x = F.relu(x)
        v = self.fc2(x)
        return v


class Actor_Critic():
    def __init__(self, dim_state: int, n_action: int, n_hidden: int, lr: float, device: str ='cpu'):
        """
        Parameter:
        ----------
        dim_state: dimension of x(s).
        n_action: number of possible actions at s.
        n_hidden: number of hidden neurons.
        lr: learning rate.
        device: 'cpu' or 'cuda' (gpu).
        """

        self.Actor = Actor(dim_state, n_action, n_hidden).to(device)
        self.Critic = Critic(dim_state, n_hidden).to(device)

        # Here we set same learning rate for Actor and Critic but you can try different learning rates for each network.
        self.Aoptimizer = torch.optim.Adam(self.Actor.parameters(), lr=lr)
        self.Coptimizer = torch.optim.Adam(self.Critic.parameters(), lr=lr)

    def choose_action(self, s: np.array) -> int:
        """
        Parameter:
        ----------
        s: state .

        Returns:
        ----------
        action: # of action.

        """
        
        s = torch.FloatTensor(s).to(device)
        a_prob = self.Actor(s)

        pi = a_prob.forward(s)
        action = np.random.choice(n_action, p=np.squeeze(pi))

        return action

    def Actor_learn(self, s: np.array, a: int, td_error: torch.tensor):
        """
        Make one step optimization

        Parameter:
        ----------
        s: state .
        a: action.
        td_error: TD error.
        """
        
        s = torch.FloatTensor(s).to(device)

        # Caveat: Optimizer will perform minimization update.
        loss = (-np.log(a) @ td_error).sum()

        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def Critic_learn(self, transition: list) -> torch.tensor:
        """
        Make one step optimization

        Parameter:
        ----------
        transition: [s,[r],[a],s_next,[done]]
        Returns:
        ----------
        td_error: TD error.

        Note: By detach operation, the returned td_error is not a function of critic parameters. 
        Gradient of Critic parameters will not be changed by backward of function of detached td_error.

        """
        
        s = torch.FloatTensor(transition[0]).to(device)
        r = transition[1][0]
        s_ = torch.FloatTensor(transition[3]).to(device)
        done = transition[4][0]

        td_error = 1 + gamma*self.forward(s_) - self.forward(s)

        loss = (td_error*td_error).sum()

        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()
        return td_error.detach()


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    env.seed(42)

    lr = 1e-3  # Learning rate
    gamma = 0.9  # Discount factor
    n_hidden = 32  # Number of Hidden Neurons
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_action = env.action_space.n
    dim_state = env.observation_space.shape[0]

    ac = Actor_Critic(dim_state, n_action, n_hidden=n_hidden, lr=lr, device=device)
    return_100 = []  # Test for solved
    return_list = []  # List for Plot

    n_trails = 1000
    for episode in range(n_trails):
        if len(return_100) == 100:
            return_100.pop(0)

        s = env.reset()
        total_return = 0
        
        while True:
            a = ac.choose_action(s)
            s_, r, done, _ = env.step(a)
            total_return += r
            transition = [s, [r], [a], s_, [done]]

            td_error = ac.Critic_learn(transition)
            ac.Actor_learn(s, a, td_error)
            if done:
                break
            s = s_
            
        return_list.append(total_return)
        return_100.append(total_return)
        
        if np.mean(return_list) > 195:
            print('Solved!')
            break
        
        if(episode % 10 == 0 and episode != 0):
            print("Episode:"+format(episode)+",score:"+format(total_return))
            

    plt.plot(return_list)
    plt.xlabel('Training Episode')
    plt.ylabel('Uncounted Return')
    plt.savefig('img/Q4_Actor_Critic.png')
    print('Average undiscounted return over the last 100 consecutive trials is %f' % (
        np.mean(return_100)))

    plt.close()
    moving_average = [np.mean(return_list[i:(i+100)]) for i in range(n_trails-100)]
    
    plt.plot(moving_average)
    plt.xlabel('Training Episode')
    plt.ylabel('Average Return')
    plt.savefig('img/Q4_Actor_Critic_Average.png')
    plt.close()