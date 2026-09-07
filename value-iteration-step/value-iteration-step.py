def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    new_values=  []
    for s in range(len(values)):
        action_values = []
        for a in range(len(transitions[s])):
            second_term = sum(
                probab * next_value
                for probab, next_value in zip(transitions[s][a], values)
            )
            action_values.append(rewards[s][a]  + gamma * second_term)
        new_values.append(max(action_values))
    return new_values