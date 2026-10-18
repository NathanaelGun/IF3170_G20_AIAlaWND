from random import Random
from src.core import Problem, SearchResult, State, is_valid, random_state, objective, sample_neighbor
from src.core.models import integer, number
from time import perf_counter
from math import exp


def solve(problem: Problem, config: dict | None = None, seed: int | None = None) -> SearchResult:
    # Baca config
    config = dict(config or {})

    unknown = set(config) - {"initial_state", "initial_temperature", "max_iterations"}
    if unknown: raise ValueError(f"Unknown SA parameters: {list(unknown)}")

    initial_temperature = config.get("initial_temperature", 100.0)
    max_iterations = config.get("max_iterations", 1000)

    number(initial_temperature, "initial_temperature")
    if initial_temperature == 0: raise ValueError("initial_temperature must be greater than 0")

    integer(max_iterations, "max_iterations", 1)

    # generate random lokal
    rng = Random(seed)

    # pengukuran waktu
    started = perf_counter()

    # state awal
    initial = config.get("initial_state")
    if initial is None: initial = random_state(problem, seed)

    if not isinstance(initial, State) or not is_valid(problem, initial): 
        raise ValueError("initial_state must be a valid State")
    initial = initial.copy()
    current = initial.copy()

    value = objective(problem, current)
    history = [{
        "iteration": 0,
        "objective": value,
        "temperature": None,
        "delta_e": None,
        "exp_delta_over_temperature": None,
        "acceptance_probability": None,
        "accepted": None,
        "stuck": None,
    }]

    iterations = 0
    stop_reason = None

    while True:
        t = iterations + 1
        temperature = initial_temperature * max (0.0, 1.0 - (t - 1) / max_iterations)

        if temperature == 0:
            stop_reason = "temperature_zero"
            break
        candidate = sample_neighbor(problem, current, rng)
        if candidate is None: 
            stop_reason = "no_valid_neighbor"
            break

        candidate_value = objective(problem, candidate)
        delta_e = candidate_value - value
        iterations += 1

        try: exp_delta_over_temperature = exp(delta_e / temperature)
        except OverflowError: exp_delta_over_temperature = None

        if delta_e > 0 : 
            acceptance_probability = 1.0
            accepted = True
        else:
            acceptance_probability = exp_delta_over_temperature
            accepted = rng.random() < acceptance_probability

        if accepted: 
            current = candidate
            value = candidate_value

        history.append({
            "iteration": iterations,
            "objective": value,
            "temperature": temperature,
            "delta_e": delta_e,
            "exp_delta_over_temperature": exp_delta_over_temperature,
            "acceptance_probability": acceptance_probability,
            "accepted": accepted,
            "stuck": None,
        })

    duration = perf_counter() - started

    return SearchResult(
        algorithm="sa",
        parameters={
            "initial_temperature": initial_temperature,
            "max_iterations": max_iterations,
            "schedule": "linear",
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
            "stuck_count": None,
        },
    )