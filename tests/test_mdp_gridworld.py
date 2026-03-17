from mdp_gridworld import GridworldMDP, compute_assignment_values


def test_non_terminal_state_count_is_9() -> None:
    mdp = GridworldMDP()
    assert len(mdp.non_terminal_states) == 9


def test_transition_distribution_for_1_2_east() -> None:
    mdp = GridworldMDP()
    dist = mdp.transition_distribution((1, 2), "East")
    assert set(dist.keys()) == {(1, 2), (1, 3), (1, 1)}
    assert dist[(1, 2)] == 0.8
    assert dist[(1, 3)] == 0.1
    assert dist[(1, 1)] == 0.1
    assert abs(sum(dist.values()) - 1.0) < 1e-12


def test_v1_values_match_hand_computation() -> None:
    mdp = GridworldMDP()
    _, v1, _ = compute_assignment_values()

    assert v1[(4, 3)] == 1.0
    assert v1[(4, 2)] == -1.0
    assert v1[(3, 3)] == 0.76

    for state in mdp.non_terminal_states:
        if state != (3, 3):
            assert v1[state] == -0.04


def test_v2_for_3_2_and_3_1() -> None:
    _, _, v2 = compute_assignment_values()
    assert abs(v2[(3, 2)] - 0.464) < 1e-12
    assert v2[(3, 1)] == -0.08
