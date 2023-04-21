import typing as t
from dataclasses import dataclass, field
import numpy as np
import logging
from rich.progress import track

from final.event import EventContainer
from final.reward_table import RewardTable
from final.agents import Agent

logger = logging.getLogger("final")


@dataclass
class SimulationResult:
    reward_averages: tuple[np.ndarray, ...]
    actions: tuple[np.ndarray, ...]


@dataclass
class Simulation(EventContainer):
    agents: t.Sequence[Agent]
    table: RewardTable

    def __post_init__(self):
        super().__init__()

    def run(self, steps: int, name: str) -> SimulationResult:
        all_actions: list[list[int]] = [[] for _ in self.agents]
        all_rewards: list[list[float]] = [[] for _ in self.agents]
        reward_averages: list[list[float]] = [[] for _ in self.agents]

        for step in track(range(steps), description=f"Running {name} Simulation"):
            self.event("step", {"step": step})
            if step % 10 == 0:
                logger.info(f"Step {step}")

            actions, rewards = self.take_step()

            for i, (action, reward) in enumerate(zip(actions, rewards)):
                all_actions[i].append(action)
                all_rewards[i].append(reward)
                reward_averages[i].append(sum(all_rewards[i]) / len(all_rewards[i]))

        return SimulationResult(
            reward_averages=tuple(np.array(x) for x in reward_averages),
            actions=tuple(np.array(x) for x in all_actions),
        )

    def take_step(self) -> tuple[t.Sequence[int], t.Sequence[int]]:
        actions = []
        for agent in self.agents:
            action = agent.take_action()
            actions.append(action)
            agent.action_counts[action] += 1

        rewards = self.table.reward(*actions)

        for agent, action, reward in zip(self.agents, actions, rewards):
            agent.update_estimate(action, reward)

        return actions, rewards
