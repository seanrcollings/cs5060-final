import typing as t
import abc
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


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

    # def run(self, steps: int, bandits: NormalBandits):
    #     actions, rewards = [], []
    #     reward_averages = []

    #     for step in range(steps):
    #         action, reward = self.take_step(step, bandits)
    #         actions.append(action)
    #         rewards.append(reward)
    #         reward_averages.append(sum(rewards) / len(rewards))

    #     return np.array(reward_averages)

    # def take_step(self, step: int, bandits: NormalBandits):
    #     action = self.take_action()
    #     self.action_counts[action] += 1
    #     reward = bandits.reward(action)
    #     self.update_estimate(action, reward)
    #     return action, reward

    @abc.abstractmethod
    def take_action(self) -> int:
        ...

    @abc.abstractmethod
    def update_estimate(self, action: int, reward: float):
        ...


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


class ThompshonSamplingAgent(Agent):
    def __init__(self, num_actions: int):
        super().__init__(num_actions)
        self._as: t.List[float] = [1] * self.num_actions
        self._bs: t.List[float] = [1] * self.num_actions
        self._As: t.List[float] = [1] * self.num_actions
        self._Bs: t.List[float] = [1] * self.num_actions

    def __str__(self) -> str:
        return f"ThompsonSamling"

    @property
    def estimates(self):
        return [
            self._As[i] / (self._As[i] + self._Bs[i]) for i in range(self.num_actions)
        ]

    @property
    def best_estimated_reward(self):
        return max(self.estimates)

    # def take_step(self, step, bandits: NormalBandits):
    #     res = super().take_step(step, bandits)
    #     if step == self.reset_after:
    #         self._as = [1] * self.num_actions
    #         self._bs = [1] * self.num_actions
    #         self._As = [1] * self.num_actions
    #         self._Bs = [1] * self.num_actions
    #     return res

    def take_action(self):
        return np.argmax(np.random.beta(self._as, self._bs))

    def update_estimate(self, action: int, reward: float):
        self._As[action] += reward
        self._Bs[action] += 1 - reward

        reward = 1 / (1 + np.exp(-reward))

        self._as[action] += reward
        self._bs[action] += 1 - reward


@dataclass
class RewardTable:
    table: dict[tuple[int, int], tuple[int, int]]

    def reward(self, act1: int, act2: int) -> tuple[int, int]:
        return self.table[(act1, act2)]


@dataclass
class Simulation:
    agent1: Agent
    agent2: Agent
    table: RewardTable

    def run(self, steps: int):
        actions1, actions2, rewards1, rewards2 = [], [], [], []
        reward_averages1, reward_averages2 = [], []

        for _ in range(steps):
            action1, action2, reward1, reward2 = self.take_step()
            actions1.append(action1)
            actions2.append(action2)
            rewards1.append(reward1)
            rewards2.append(reward2)
            reward_averages1.append(sum(rewards1) / len(rewards1))
            reward_averages2.append(sum(rewards2) / len(rewards2))

        return (
            np.array(reward_averages1),
            np.array(reward_averages2),
            np.array(actions1),
            np.array(actions2),
        )

    def take_step(self) -> tuple[int, int, int, int]:
        action1 = self.agent1.take_action()
        self.agent1.action_counts[action1] += 1
        action2 = self.agent2.take_action()
        self.agent2.action_counts[action2] += 1

        reward1, reward2 = self.table.reward(action1, action2)

        self.agent1.update_estimate(action1, reward1)
        self.agent2.update_estimate(action2, reward2)

        return action1, action2, reward1, reward2


def run_simulation():
    steps = 1000
    epsilon = 0.1

    agent1 = EpsilonGreedyAgent(2, epsilon)
    agent2 = ThompshonSamplingAgent(2)
    table = RewardTable(
        {
            (0, 0): (-1, -1),
            (0, 1): (4, -8),
            (1, 0): (-8, 4),
            (1, 1): (-4, -4),
        }
    )

    simulation = Simulation(agent1, agent2, table)
    reward_averages1, reward_averages2, actions1, actions2 = simulation.run(steps)

    plt.bar(
        (0, 1),
        (
            np.count_nonzero(
                actions1,
            )
        ),
        label="Epsilon Greedy",
    )
    plt.bar(actions2, label="Thompson Sampling")

    plt.legend()
    plt.show()


run_simulation()
