# https://www.youtube.com/watch?v=__NYgfkUr-M&t=119s

import collections
import random

import torch
import torch.nn as nn

# import environment
import gym

class ReplayBuffer():
    def __init__(self):
        """
        FIFO
        """
        self.buffer = collections.deque()
        self.batch_size = 32
        self.size_limit = 50000  # buffer의 최대 크기 (DQN 문제에 따라 크기 다름)

    def put(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()

    def sample(self, n):
        return random.sample(self.buffer, n)

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
    # env = environment.Environment()
    q = Qnet(actions=2)
    q_target = Qnet(actions=2)
    q_target.load_state_dict(q.state_dict()) #q model의 weight 정보를 dictionary 형태로 닮고 있음
    memory = ReplayBuffer()

    avg_t = 0
    gamma = 0.98
    batch_size = 32
    optimizer = torch.optim.Adam(q.parameters(), lr=0.0005)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))
        state = env.reset()

        for t in range(600):
            action = q.sample_action(obs=torch.from_numpy(state).float(), epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward/200.0, next_state, done_mask))
            state = next_state

            if done:
                break

    if memory.size()>2000:
        # memory 쌓인 이후 학습 시작
        train(q, q_target, memory, gamma, optimizer, batch_size)

    if n_epi%20 == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())
        print("# of episode : {}, Avg timestep : {:.1f}, buffer_size : {}, epsilon : {:.1f}%".format(n_epi,
                                                                                                     avg_t/20.0,
                                                                                                     memory.size(),
                                                                                                     epsilon*100))

def train(q, q_target, memory, gamma, optimizer, batch_size):
    for i in range(10):
        batch = memory.sample(batch_size)
        state_list, action_list, reward_list, next_state_list, done_mask_list = [], [], [], [], []

        for transition in batch:
            state, action, reward, next_state, done_mask = transition
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_mask_list.append(done_mask)

        state, action, reward, next_state, done_mask = torch.tensor(state_list, dtype=torch.float), \
                                                       torch.tensor(action_list), \
                                                       torch.tensor(reward_list), \
                                                       torch.tensor(next_state_list, dtype=torch.float), \
                                                       torch.tensor(done_mask_list)

        q_out = q(state)               # Shape : [32, 2]
        q_a = q_out.gather(1, action)  # 취한 action의 q 값만 골라냄. Shape : [32, 1]
        max_q_prime = q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + gamma * max_q_prime * done_mask
        loss = torch.nn.functional.smooth_l1_loss(target, q_a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

