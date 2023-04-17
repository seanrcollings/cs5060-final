import abc
import typing as t

import numpy as np
import torch
import torch.nn as nn


class Agent(abc.ABC):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions)

    @property
    @abc.abstractmethod
    def estimates(self):
        ...

    @property
    @abc.abstractmethod
    def best_estimated_reward(self):
        ...

    @abc.abstractmethod
    def take_action(self) -> int:
        ...

    @abc.abstractmethod
    def update_estimate(self, action: int, reward: float):
        ...


class StaticAgent(Agent):
    def __init__(self, num_actions: int, action: int):
        super().__init__(num_actions)
        self.action = action
        self._estimate = np.zeros(self.num_actions)

    def __str__(self) -> str:
        return f"StaticAgent({self.action})"

    @property
    def estimates(self):
        return self._estimate

    def best_estimated_reward(self):
        return self._estimate[self.action]

    def update_estimate(self, action: int, reward: float):
        self._estimate[action] = reward

    def take_action(self):
        return self.action


class EpsilonGreedyAgent(Agent):
    def __init__(self, num_actions: int, epsilon: float):
        super().__init__(num_actions)
        self.epsilon = epsilon
        self._estimates = np.zeros(num_actions)

    def __str__(self) -> str:
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    @property
    def estimates(self):
        return self._estimates

    @property
    def best_estimated_reward(self):
        return max(self._estimates)

    def update_estimate(self, action: int, reward: float):
        self._estimates[action] += (1.0 / self.action_counts[action]) * (
            reward - self._estimates[action]
        )

    def take_action(self):
        return (
            np.random.randint(self.num_actions)
            if np.random.random() < self.epsilon
            else np.random.choice(
                np.flatnonzero(self.estimates == self.estimates.max())
            )
        )


class ThompsonSamplingAgent(Agent):
    def __init__(self, num_actions: int):
        super().__init__(num_actions)
        self._as: t.List[float] = [1] * self.num_actions
        self._bs: t.List[float] = [1] * self.num_actions
        self._As: t.List[float] = [1] * self.num_actions
        self._Bs: t.List[float] = [1] * self.num_actions

    def __str__(self) -> str:
        return f"ThompsonSampling"

    @property
    def estimates(self):
        return [
            self._As[i] / (self._As[i] + self._Bs[i]) for i in range(self.num_actions)
        ]

    @property
    def best_estimated_reward(self):
        return max(self.estimates)

    def take_action(self):
        return np.argmax(np.random.beta(self._as, self._bs))

    def update_estimate(self, action: int, reward: float):
        self._As[action] += reward
        self._Bs[action] += 1 - reward

        reward = 1 / (1 + np.exp(-reward))

        self._as[action] += reward
        self._bs[action] += 1 - reward


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


class NetworkAgent(Agent):
    def __init__(
        self,
        num_actions: int,
        network: Network,
        lr: float,
    ):
        super().__init__(num_actions)

        self.lr = lr
        self.x = np.array([0, 1])
        self._estimates = np.random.random(num_actions)

        self.model = network

    def __str__(self) -> str:
        return f"NetworkAgent(lr={self.lr})"

    @property
    def estimates(self):
        return self._estimates

    @property
    def best_estimated_reward(self):
        return np.max(self._estimates)

    def fit(self):
        x = torch.from_numpy(self.x).float()
        y = torch.from_numpy(self._estimates).float()

        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        outputs = self.model(x)
        loss = self.criterion(outputs.reshape(2), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def take_action(self):
        self.fit()
        x = torch.from_numpy(self.x).float()
        y_hat = self.model(x)
        action = np.argmax(y_hat.detach().numpy())
        return action

    def update_estimate(self, action: int, reward: float):
        self._estimates[action] += (1.0 / self.action_counts[action]) * (
            reward - self._estimates[action]
        )
