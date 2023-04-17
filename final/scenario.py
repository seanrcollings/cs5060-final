from dataclasses import dataclass
import logging
import typing as t

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from final.reward_table import RewardTable
from final.agents import Agent
from final.simulation import Simulation, SimulationResult

logger = logging.getLogger("final")


@dataclass
class Scenario:
    name: str
    steps: int
    agent1: Agent
    agent2: Agent
    table: RewardTable

    def run(self, plot: bool = False, verbose: bool = False):
        simulation = Simulation(self.agent1, self.agent2, self.table)

        if verbose:
            logging.basicConfig(level=logging.INFO)

        res = simulation.run(self.steps)

        if plot:
            self.plot_simulation(res)

    def plot_simulation(self, res: SimulationResult):
        actions1, actions2 = res.actions
        # Bar Graphs

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2)

        ax_greedy = fig.add_subplot(gs[0, 0])
        ax_thompson = fig.add_subplot(gs[0, 1])

        unique_1, counts_1 = np.unique(actions1, return_counts=True)
        unique_2, counts_2 = np.unique(actions2, return_counts=True)

        fig.suptitle(f"{self.name}: {self.agent1} vs {self.agent2}")

        self.plot_bar(ax_greedy, unique_1, counts_1, str(self.agent1))
        self.plot_bar(ax_thompson, unique_2, counts_2, str(self.agent2))

        # Line Graph
        reward_averages1, reward_averages2 = res.reward_averages

        ax_lines = fig.add_subplot(gs[1, :])

        ax_lines.plot(reward_averages1, label=str(self.agent1))
        ax_lines.plot(reward_averages2, label=str(self.agent2))
        ax_lines.set_title("Average Reward")
        ax_lines.set_ylabel("Average Reward")
        ax_lines.set_xlabel("Step")
        ax_lines.legend()

        plt.show()

    def plot_bar(self, ax, unique, counts, title):
        spacing = 0.09

        ax.bar(unique, counts)
        ax.set_xticks(unique)

        for i in range(len(unique)):
            ax.text(i, counts[i] + spacing, counts[i], ha="center", va="bottom")

        ax.set_title(title)
