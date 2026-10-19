from math import exp
from random import Random
from time import perf_counter

from src.core import Problem, SearchResult, State, get_objective, is_valid, random_neighbor, random_state
from src.core.models import integer, number

STAGNATION_K = 50


def solve(problem: Problem, config: dict | None = None, seed: int | None = None) -> SearchResult:
    config = dict(config or {})

    unknown = set(config) - {"initial_state", "initial_temperature", "max_iterations", "objective"}
    if unknown:
        raise ValueError(f"Unknown SA parameters: {sorted(unknown)}")

    objective_name = config.get("objective", "value")
    score = get_objective(objective_name)
    initial_temperature = config.get("initial_temperature", 100.0)
    max_iterations = config.get("max_iterations", 1000)

    number(initial_temperature, "initial_temperature")
    if initial_temperature == 0:
        raise ValueError("initial_temperature must be greater than 0")

    integer(max_iterations, "max_iterations", 1)

    rng = Random(seed)
    started = perf_counter()

    initial = config.get("initial_state")
    if initial is None:
        initial = random_state(problem, seed)

    if not isinstance(initial, State) or not is_valid(problem, initial):
        raise ValueError("initial_state must be a valid State")

    initial = initial.copy()
    current = initial.copy()
    value = score(problem, current)

    best_value = value
    no_improve = 0
    stuck_count = 0

    history = [{
        "iteration": 0,
        "objective": value,
        "temperature": None,
        "delta_e": None,
        "exp_delta_over_temperature": None,
        "acceptance_probability": None,
        "accepted": None,
        "candidate_found": True,
        "stuck": False,
    }]

    iterations = 0
    stop_reason = ""

    while True:
        t = iterations + 1
        temperature = initial_temperature * max(0.0, 1.0 - (t - 1) / max_iterations)

        if temperature == 0:
            stop_reason = "temperature_zero"
            break

        candidate = random_neighbor(problem, current, rng)
        if candidate is None:
            iterations += 1
            no_improve += 1
            is_stuck = no_improve >= STAGNATION_K
            if no_improve == STAGNATION_K:
                stuck_count += 1
            history.append({
                "iteration": iterations,
                "objective": value,
                "temperature": temperature,
                "delta_e": None,
                "exp_delta_over_temperature": None,
                "acceptance_probability": None,
                "accepted": False,
                "candidate_found": False,
                "stuck": is_stuck,
            })
            continue

        candidate_value = score(problem, candidate)
        delta_e = candidate_value - value
        iterations += 1

        if delta_e >= 0:
            exp_delta_over_temperature = 1.0
            acceptance_probability = 1.0
            accepted = True
        else:
            try:
                exp_delta_over_temperature = exp(delta_e / temperature)
            except OverflowError:
                exp_delta_over_temperature = 0.0
            acceptance_probability = exp_delta_over_temperature
            accepted = rng.random() < acceptance_probability

        if accepted:
            current = candidate
            value = candidate_value

        if value > best_value:
            best_value = value
            no_improve = 0
            is_stuck = False
        else:
            no_improve += 1
            is_stuck = no_improve >= STAGNATION_K
            if no_improve == STAGNATION_K:
                stuck_count += 1

        history.append({
            "iteration": iterations,
            "objective": value,
            "temperature": temperature,
            "delta_e": delta_e,
            "exp_delta_over_temperature": exp_delta_over_temperature,
            "acceptance_probability": acceptance_probability,
            "accepted": accepted,
            "candidate_found": True,
            "stuck": is_stuck,
        })

    duration = perf_counter() - started

    return SearchResult(
        algorithm="sa",
        parameters={
            "initial_temperature": initial_temperature,
            "max_iterations": max_iterations,
            "schedule": "linear",
            "objective": objective_name,
        },
        seed=seed,
        initial_state=initial,
        final_state=current.copy(),
        final_objective=value,
        duration_seconds=duration,
        iterations=iterations,
        history=history,
        stop_reason=stop_reason,
        metadata={
            "stuck_count": stuck_count,
            "stagnation_k": STAGNATION_K,
        },
    )
