from dataclasses import dataclass


@dataclass
class RewardTable:
    table: dict[tuple[int, ...], tuple[int, ...]]

    def reward(self, *actions: int) -> tuple[int, ...]:
        return self.table[actions]
