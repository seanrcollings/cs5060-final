# Final Project

This repo contains the code for the final project of the course "Decision Making Under Uncertainty" at Utah State University.

## Installation
Install dependencies with
```
$ pip install -r requirements.txt
```

## Usage
The application comes with a command line interface. See the possible options with
```
$ python -m final -h
```

The application can run several different scenarios with different parameters. You can run a scenario with
```
$ python -m final <scenario>
```

The list of available scenarios is given in the help message.

## Design
The application is designed to be modular. The main components are the following:
- `Scenario`: a scenario is a set of parameters that define the simulation to execute. It is responsible for generating the initial state of the problem and for generating the next state given an action. It is capable of responding to events that happen during the simulation. Additionally, it is responsible for generating helpful output in the form of console output and plots.
- `Simulation`: a simulation handles the execution of an instance of a game and is responsible for keeping track of the state of the game. It is responsible for executing the game and for generating the output of the game.
- `Agents`: agents are the players of the game. They are responsible for choosing an action their current understanding of the game. They are also responsible for updating their understanding of the game based on a provided reward for each action they take.
- `RewardTable`: a reward table is a table that maps actions to rewards. It is used to calculate the rewards for a set of agent actions.
