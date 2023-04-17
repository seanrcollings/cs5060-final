from dataclasses import dataclass
import numpy as np
import logging

from final.reward_table import RewardTable
from final.agents import Agent

logger = logging.getLogger("final")


@dataclass
class SimulationResult:
    reward_averages: tuple[np.ndarray, ...]
    actions: tuple[np.ndarray, ...]


@dataclass
class Simulation:
    agent1: Agent
    agent2: Agent
    table: RewardTable

    def run(self, steps: int) -> SimulationResult:
        actions1, actions2, rewards1, rewards2 = [], [], [], []
        reward_averages1, reward_averages2 = [], []

        for step in range(steps):
            if step % 10 == 0:
                logger.info(f"Step {step}")

            action1, action2, reward1, reward2 = self.take_step()
            actions1.append(action1)
            actions2.append(action2)
            rewards1.append(reward1)
            rewards2.append(reward2)
            reward_averages1.append(sum(rewards1) / len(rewards1))
            reward_averages2.append(sum(rewards2) / len(rewards2))

        return SimulationResult(
            (
                np.array(reward_averages1),
                np.array(reward_averages2),
            ),
            (
                np.array(actions1),
                np.array(actions2),
            ),
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
