import argparse
import sys

from final.reward_table import RewardTable
from final.agents import EpsilonGreedyAgent
from final.scenario import Scenario


table = RewardTable(
    {
        (0, 0): (-1, -1),
        (0, 1): (4, -8),
        (1, 0): (-8, 4),
        (1, 1): (-4, -4),
    }
)


sc1 = Scenario(
    "Basic Scenario",
    100,
    EpsilonGreedyAgent(2, 0.1),
    EpsilonGreedyAgent(2, 0.1),
    table,
)

scenarios = {
    "sc1": sc1,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario",
        help=f"The scenario to run. Possible options: {', '.join(scenarios.keys())}",
    )
    parser.add_argument(
        "--plot", "-p", action="store_false", help="Plot the results", default=True
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
