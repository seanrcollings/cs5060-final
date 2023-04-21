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


@switch_table.on("step")
def switch_table_step(simulation, data):
    if data["step"] == 500:
        simulation.table = t2


scenarios = {
    "best_epsilon": best_epsilon,
    "best_thompson": best_thompson,
    "best_nn": best_nn,
    "switch_table": switch_table,
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
        help="Do no plot the results",
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
