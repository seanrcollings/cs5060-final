import typing as t
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from rich.panel import Panel
from rich.console import Group
from rich.table import Table

from final.event import EventContainer
from final.reward_table import RewardTable
from final.agents import Agent
from final.simulation import Simulation, SimulationResult
from final.console import console


@dataclass
class Scenario(EventContainer):
    name: str
    description: str
    steps: int
    agents: t.Sequence[Agent]
    table: RewardTable

    def __post_init__(self):
        super().__init__()

    def run(self, plot: bool = False):

        simulation = Simulation(self.agents, self.table)
        simulation.event_handlers = self.event_handlers

        res = simulation.run(self.steps, self.name)

        table = Table()
        table.add_column("Agent")
        table.add_column("Average Reward", justify="right")
        for i in np.unique(res.actions[0]):
            table.add_column(f"Action # {i} ", justify="right")

        for agent, reward, actions in zip(
            self.agents, res.reward_averages, res.actions
        ):
            _, counts = np.unique(actions, return_counts=True)

            table.add_row(
                str(agent),
                f"{reward[-1]:.2f}",
                *[str(c) for c in counts],
            )

        contents = f"[bold]Description: [/bold]{self.description}\n\n"
        contents += f"[bold]Parameters[/bold]\n"
        contents += f"  - Steps: {self.steps}\n"
        contents += f"  - Agents: {', '.join(str(a) for a in self.agents)}\n"
        contents += f"  - Reward Table: {self.table}\n"

        contents += f"\n[bold]Results[/bold]"

        group = Group(contents, table)

        console.print(Panel.fit(group, title=f"Scenario {self.name}"))

        if plot:
            self.plot_simulation(res)

        print()

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
