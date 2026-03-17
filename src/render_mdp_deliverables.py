"""Print key outputs for problem 4.4 in the uploaded solution format."""

from mdp_gridworld import GridworldMDP, compute_assignment_values


def main() -> None:
    mdp = GridworldMDP()
    v0, v1, v2 = compute_assignment_values()

    print("problem 4.4: Formulating an MDP")
    print()
    print("Part 1")
    print(f"Total states (excluding wall): {len(mdp.all_states)}")
    print(f"Non-terminal states: {len(mdp.non_terminal_states)}")
    print(f"Terminal states: {sorted(mdp.terminal_rewards.items())}")
    print()

    print("Part 2")
    dist = mdp.transition_distribution((1, 2), "East")
    print("T(s' | (1,2), East):")
    for state, prob in sorted(dist.items()):
        print(f"  {state}: {prob:.1f}")
    print()

    print("Part 4")
    print(f"V1(3,3) = {v1[(3, 3)]:.2f}")
    print(f"V2(3,2) = {v2[(3, 2)]:.3f}")
    print(f"V2(3,1) = {v2[(3, 1)]:.2f}")
    print()

    print("Part 5 policy sketch")
    print("+-----+-----+-----+------+")
    print("|  -> |  -> |  -> | GOAL |")
    print("+-----+-----+-----+------+")
    print("|  ^  | WALL|  ^  | HAZD |")
    print("+-----+-----+-----+------+")
    print("|  ^  |  -> |  ^  |  <-  |")
    print("+-----+-----+-----+------+")

    _ = v0  # keep visible in script context for quick debugging if needed


if __name__ == "__main__":
    main()
