from dataclasses import dataclass


@dataclass
class RewardTable:
    table: dict[tuple[int, int], tuple[int, int]]

    def reward(self, act1: int, act2: int) -> tuple[int, int]:
        return self.table[(act1, act2)]
