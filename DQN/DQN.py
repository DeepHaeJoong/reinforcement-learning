# https://www.youtube.com/watch?v=__NYgfkUr-M&t=119s

# libraries
import gym
import collections
import random

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000        # size of replay buffer
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        """
        FIFO
        """
        self.buffer = collections.deque(maxlen=buffer_limit)  # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, actions=2):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def main():
    env = gym.make('CartPole-v1')

    q = Qnet(actions=2)          # Initialize the main network
    q_target = Qnet(actions=2)   # Initialize the target network
    q_target.load_state_dict(q.state_dict()) #q model의 weight 정보를 dictionary 형태로 닮고 있음
    memory = ReplayBuffer()      # Initialize the replay buffer D

    score = 0
    time_step = 2000
    optimizer = torch.optim.Adam(q.parameters(), lr=0.0005)  # the main parameter

    # For N number of epsisodes
    for n_epi in range(2000):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))   # 0.08 -> 0.01
        state = env.reset()

        for t in range(time_step):
            # observe the state (s) and select action using epsilon-greedy policy.
            action = q.sample_action(obs=torch.from_numpy(state).float(), epsilon=epsilon)
            # perform the selected action and move the next state (s') and obtain the reward (r)
            next_state, reward, done, info = env.step(action)
            done_mask = 0.0 if done else 1.0
            # store the transition information of K (600) transitions from the replay buffer D
            memory.put((state, action, reward/200.0, next_state, done_mask))
            state = next_state

            score += reward

            if done:
                break

        if memory.size() > 2000:
            # memory 쌓인 이후 학습 시작
            train(q, q_target, memory, gamma, optimizer, batch_size)

        if n_epi%20 == 0 and n_epi != 0:
            # Freeze the target network parameter (several time step) and then update it by just copying the main network parameter
            q_target.load_state_dict(q.state_dict())
            print("# of episode : {}, score : {:.1f}, buffer_size : {}, epsilon : {:.1f}%".format(n_epi,
                                                                                                  score/20.0,
                                                                                                  memory.size(),
                                                                                                  epsilon*100))
            score = 0.0

def train(q, q_target, memory, gamma, optimizer, batch_size):

    for i in range(10):
        # Randomly sample a mini-batch of K transitions from the replay buffer D
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)  # 취한 action의 q 값만 골라냄. Shape : [32, 1]
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        # Compute the loss and the gradient of the loss and update main
        # MSE Loss
        loss = F.mse_loss(q_a, target)

        # Smooth L1 Loss
        #loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

