"""Gridworld MDP formulation for the 4x3 robot problem using (x, y) states.

Coordinates follow the assignment convention: (1,1) is bottom-left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

State = Tuple[int, int]
Action = str


@dataclass(frozen=True)
class GridworldMDP:
    """4x3 stochastic gridworld with one wall and two terminal states."""

    n_cols: int = 4
    n_rows: int = 3
    wall: State = (2, 2)
    terminal_rewards: Dict[State, float] = None  # type: ignore[assignment]
    default_reward: float = -0.04

    def __post_init__(self) -> None:
        if self.terminal_rewards is None:
            object.__setattr__(self, "terminal_rewards", {(4, 3): 1.0, (4, 2): -1.0})

    @property
    def action_space(self) -> Tuple[Action, Action, Action, Action]:
        return ("North", "South", "East", "West")

    @property
    def all_states(self) -> List[State]:
        return [
            (x, y)
            for y in range(1, self.n_rows + 1)
            for x in range(1, self.n_cols + 1)
            if (x, y) != self.wall
        ]

    @property
    def non_terminal_states(self) -> List[State]:
        return [s for s in self.all_states if s not in self.terminal_rewards]

    def reward(self, state: State) -> float:
        return self.terminal_rewards.get(state, self.default_reward)

    def is_inside(self, state: State) -> bool:
        x, y = state
        return 1 <= x <= self.n_cols and 1 <= y <= self.n_rows

    def move(self, state: State, action: Action) -> State:
        if state in self.terminal_rewards:
            return state

        x, y = state
        deltas = {
            "North": (0, 1),
            "South": (0, -1),
            "East": (1, 0),
            "West": (-1, 0),
        }
        dx, dy = deltas[action]
        candidate = (x + dx, y + dy)
        if (not self.is_inside(candidate)) or candidate == self.wall:
            return state
        return candidate

    def transition_distribution(self, state: State, action: Action) -> Dict[State, float]:
        """Return T(s'|s,a) using 0.8 intended and 0.1 each perpendicular."""
        if state in self.terminal_rewards:
            return {state: 1.0}

        perpendicular = {
            "North": ("West", "East"),
            "South": ("West", "East"),
            "East": ("North", "South"),
            "West": ("North", "South"),
        }

        outcomes = [
            (self.move(state, action), 0.8),
            (self.move(state, perpendicular[action][0]), 0.1),
            (self.move(state, perpendicular[action][1]), 0.1),
        ]

        merged: Dict[State, float] = {}
        for s_next, prob in outcomes:
            merged[s_next] = merged.get(s_next, 0.0) + prob
        return merged

    def bellman_backup(self, values: Dict[State, float], state: State, gamma: float = 1.0) -> float:
        if state in self.terminal_rewards:
            return self.reward(state)

        action_values: List[float] = []
        for action in self.action_space:
            dist = self.transition_distribution(state, action)
            expected = sum(prob * values[s_next] for s_next, prob in dist.items())
            action_values.append(expected)

        return self.reward(state) + gamma * max(action_values)

    def value_iteration_step(self, values: Dict[State, float], gamma: float = 1.0) -> Dict[State, float]:
        next_values: Dict[State, float] = {}
        for state in self.all_states:
            next_values[state] = self.bellman_backup(values, state, gamma=gamma)
        return next_values

    def initial_values(self) -> Dict[State, float]:
        values: Dict[State, float] = {}
        for state in self.all_states:
            values[state] = 0.0 if state not in self.terminal_rewards else self.reward(state)
        return values


def format_distribution(dist: Dict[State, float]) -> List[Tuple[State, float]]:
    """Stable ordering helper for report-like output."""
    return sorted(dist.items(), key=lambda item: (item[0][0], item[0][1]))


def compute_assignment_values() -> tuple[Dict[State, float], Dict[State, float], Dict[State, float]]:
    """Return (V0, V1, V2) exactly as required by the assignment setup."""
    mdp = GridworldMDP()
    v0 = mdp.initial_values()
    v1 = mdp.value_iteration_step(v0, gamma=1.0)
    v2 = mdp.value_iteration_step(v1, gamma=1.0)
    return v0, v1, v2
