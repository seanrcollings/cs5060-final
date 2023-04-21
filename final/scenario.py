import typing as t
from dataclasses import dataclass
import logging

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from final.event import EventContainer
from final.reward_table import RewardTable
from final.agents import Agent
from final.simulation import Simulation, SimulationResult

logger = logging.getLogger("final")


@dataclass
class Scenario(EventContainer):
    name: str
    steps: int
    agents: t.Sequence[Agent]
    table: RewardTable

    def __post_init__(self):
        super().__init__()

    def run(self, plot: bool = False, verbose: bool = False):
        print(f"Running Scenario {self.name}")
        print(f"  Running for {self.steps} steps")
        print(f"  Agents: {', '.join(str(a) for a in self.agents)}")
        print(f"  Reward Table: {self.table}")

        simulation = Simulation(self.agents, self.table)
        simulation.event_handlers = self.event_handlers

        if verbose:
            logging.basicConfig(level=logging.INFO)

        res = simulation.run(self.steps)

        if plot:
            self.plot_simulation(res)

    def plot_simulation(self, res: SimulationResult):
        # Bar Graphs

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"{self.name}: {' vs '.join(str(a) for a in self.agents)}")
        gs = fig.add_gridspec(2, len(self.agents))

        for idx, actions in enumerate(res.actions):
            ax = fig.add_subplot(gs[0, idx])
            unique, counts = np.unique(actions, return_counts=True)
            self.plot_bar(ax, unique, counts, str(self.agents[idx]))

        # Line Graph

        ax_lines = fig.add_subplot(gs[1, :])
        ax_lines.set_title("Average Reward")
        ax_lines.set_ylabel("Average Reward")
        ax_lines.set_xlabel("Step")

        for idx, rewards in enumerate(res.reward_averages):
            ax_lines.plot(rewards, label=str(self.agents[idx]))

        ax_lines.legend()

        plt.show()

    def plot_bar(self, ax, unique, counts, title):
        spacing = 0.09

        ax.bar(unique, counts)
        ax.set_xticks(unique)

        for i in range(len(unique)):
            ax.text(i, counts[i] + spacing, counts[i], ha="center", va="bottom")

        ax.set_title(title)
