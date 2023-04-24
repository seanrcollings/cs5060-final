import argparse
import sys
import numpy as np
import torch

from final.reward_table import RewardTable
from final.agents import (
    EpsilonGreedyAgent,
    Network,
    NetworkAgent,
    ThompsonSamplingAgent,
)
from final.scenario import Scenario

np.random.seed(1231412)  # Set the seed for reproducibility
torch.seed()

table = RewardTable(
    {
        (0, 0): (-1, -1),
        (0, 1): (4, -8),
        (1, 0): (-8, 4),
        (1, 1): (-4, -4),
    }
)

t2 = RewardTable(
    {
        (0, 0): (-1, -20),
        (0, 1): (-4, 20),
        (1, 0): (8, -20),
        (1, 1): (4, 20),
    }
)


three_agents_table = RewardTable(
    {
        (0, 0, 0): (7, 7, 7),
        (0, 0, 1): (3, 3, 9),
        (0, 1, 0): (3, 9, 3),
        (0, 1, 1): (9, 3, 3),
        (1, 0, 0): (0, 5, 5),
        (1, 0, 1): (5, 0, 5),
        (1, 1, 0): (5, 5, 0),
        (1, 1, 1): (1, 1, 1),
    }
)

# Where each algorithm is the best of the competition
best_epsilon = Scenario(
    "Best Epsilon",
    "Scenario where Epsilon Greedy is the best",
    steps=100,
    agents=(EpsilonGreedyAgent(2, 0.1), ThompsonSamplingAgent(2)),
    table=table,
)

best_thompson = Scenario(
    "Best Thompson",
    "Scenario where Thompson is the best",
    steps=1000,
    agents=(EpsilonGreedyAgent(2, 0.1), ThompsonSamplingAgent(2)),
    table=table,
)

best_nn = Scenario(
    "Best NN",
    "Scenario where the Neural Network is the best",
    steps=1000,
    agents=(
        EpsilonGreedyAgent(2, 0.1),
        NetworkAgent(2, Network(2, 2), 0.1),
    ),
    table=table,
)

switch_table_no_nn = Scenario(
    "Switch Table",
    "Scenario where the table is replaced with a new one after 500 steps",
    steps=5000,
    agents=(EpsilonGreedyAgent(2, 0.1), ThompsonSamplingAgent(2)),
    table=table,
)

switch_table = Scenario(
    "Switch Table",
    "Scenario where the table is replaced with a new one after 500 steps",
    steps=10000,
    agents=(
        EpsilonGreedyAgent(2, 0.1),
        NetworkAgent(2, Network(2, 2), 0.1),
    ),
    table=table,
)

three_agents = Scenario(
    "Three Agents",
    "Scenario where there are three agents",
    steps=1000,
    agents=(
        EpsilonGreedyAgent(2, 0.1),
        ThompsonSamplingAgent(2),
        NetworkAgent(2, Network(2, 2), 0.1),
    ),
    table=three_agents_table,
)


@switch_table.on("step")
def switch_table_step(simulation, data):
    if data["step"] == 500:
        simulation.table = t2


@switch_table_no_nn.on("step")
def switch_table_step(simulation, data):
    if data["step"] == 500:
        simulation.table = t2


scenarios = {
    "best_epsilon": best_epsilon,
    "best_thompson": best_thompson,
    "best_nn": best_nn,
    "switch_table_no_nn": switch_table_no_nn,
    "switch_table": switch_table,
    "three_agents": three_agents,
}


def main():
    parser = argparse.ArgumentParser("final")
    parser.add_argument(
        "scenario",
        help=(
            "The scenario to run. Note that 'all' will runn all other scenarios. "
            f"Possible options: all, {', '.join(scenarios.keys())}"
        ),
    )
    parser.add_argument(
        "--no-plot",
        "-n",
        dest="plot",
        action="store_false",
        help="Do not plot the results",
        default=True,
    )

    res = parser.parse_args(sys.argv[1:])

    if res.scenario == "all":
        for scenario in scenarios.values():
            scenario.run(res.plot)
    elif res.scenario not in scenarios:
        print(f"Invalid scenario: {res.scenario}")
        sys.exit(1)
    else:
        scenarios[res.scenario].run(res.plot)


main()
