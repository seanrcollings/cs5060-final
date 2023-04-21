import argparse
import sys
import numpy as np

from final.reward_table import RewardTable
from final.agents import (
    EpsilonGreedyAgent,
    Network,
    NetworkAgent,
    ThompsonSamplingAgent,
)
from final.scenario import Scenario

np.random.seed(1231412)  # Set the seed for reproducibility

table = RewardTable(
    {
        (0, 0): (-1, -1),
        (0, 1): (4, -8),
        (1, 0): (-8, 4),
        (1, 1): (-4, -4),
    }
)


best_epsilon = Scenario(
    "Best Epsilon",
    steps=100,
    agents=(EpsilonGreedyAgent(2, 0.1), ThompsonSamplingAgent(2)),
    table=table,
)

best_thompson = Scenario(
    "Best Thompson",
    steps=1000,
    agents=(EpsilonGreedyAgent(2, 0.1), ThompsonSamplingAgent(2)),
    table=table,
)

best_nn = Scenario(
    "Best NN",
    steps=1000,
    agents=(
        EpsilonGreedyAgent(2, 0.1),
        NetworkAgent(2, Network(2, 2), 0.1),
    ),
    table=table,
)

scenarios = {
    "best_epsilon": best_epsilon,
    "best_thompson": best_thompson,
    "best_nn": best_nn,
}


def main():
    parser = argparse.ArgumentParser("final")
    parser.add_argument(
        "scenario",
        help=f"The scenario to run. Possible options: {', '.join(scenarios.keys())}",
    )
    parser.add_argument(
        "--no-plot",
        "-n",
        dest="plot",
        action="store_false",
        help="Do no plot the results",
        default=True,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print out logging messages",
        default=False,
    )

    res = parser.parse_args(sys.argv[1:])

    if res.scenario not in scenarios:
        print(f"Invalid scenario: {res.scenario}")
        sys.exit(1)

    scenarios[res.scenario].run(res.plot, res.verbose)


main()
