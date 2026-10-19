from math import exp, isfinite
from random import Random
from time import perf_counter

from src.core import (Problem, SearchResult, State, get_objective, is_valid,
                      move_package, random_neighbor, random_state, swap_packages)
from src.core.geometry import oriented_dimensions
from src.core.models import integer, number


def _has_improving_neighbor(problem: Problem, state: State, objective_name: str) -> bool:
    """Check the exact improving neighborhood for the two additive objectives.

    Rotation and moving already loaded packages preserve the loaded package set.
    Only insertion or an outside/inside swap can increase its score. Insertion
    keeps the current orientation and needs either the floor or a package top.
    """
    score = get_objective(objective_name)
    value = score(problem, state)
    inside = [p for p in problem.packages if state.placements[p.id].inside]
    outside = [p for p in problem.packages if not state.placements[p.id].inside]
    contributions = {p.id: p.value if objective_name == "value" else p.value / (1 + p.eta)
                     for p in problem.packages}
    weights = [sum(p.weight for p in inside if state.placements[p.id].truck_index == index)
               for index in range(len(problem.trucks))]
    for package in outside:
        if contributions[package.id] <= 0:
            continue
        w, l, h = oriented_dimensions(package, state.placements[package.id])
        for index, truck in enumerate(problem.trucks):
            if weights[index] + package.weight > truck.max_capacity:
                continue
            d = truck.dimensions
            support_heights = {0}
            support_heights.update(
                state.placements[p.id].position[2] + oriented_dimensions(p, state.placements[p.id])[2]
                for p in inside if not p.is_fragile and state.placements[p.id].truck_index == index)
            for z in sorted(support_heights):
                if z + h > d.h:
                    continue
                for y in range(d.l - l + 1):
                    for x in range(d.w - w + 1):
                        candidate = move_package(state, package.id, (x, y, z), index)
                        if is_valid(problem, candidate) and score(problem, candidate) > value:
                            return True
        for loaded in inside:
            if contributions[package.id] <= contributions[loaded.id]:
                continue
            candidate = swap_packages(state, package.id, loaded.id)
            if is_valid(problem, candidate) and score(problem, candidate) > value:
                return True
    return False


def _exponential_metrics(delta_e: float, temperature: float) -> tuple[float | None, float | str, bool]:
    """Keep the unclipped exponential and its logarithm JSON serializable."""
    exponent = delta_e / temperature
    try:
        raw = exp(exponent)
    except OverflowError:
        raw = None
    overflow = raw is None or not isfinite(raw)
    if overflow:
        raw = None
    if not isfinite(exponent):
        # Preserve extreme ratios without putting Infinity in saved JSON.
        from decimal import Decimal
        exponent = str(Decimal(str(delta_e)) / Decimal(str(temperature)))
    return raw, exponent, overflow


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

    local_optima = {}

    def is_local_optimum(state: State) -> bool:
        key = tuple(state.placements[p.id] for p in problem.packages)
        if key not in local_optima:
            local_optima[key] = not _has_improving_neighbor(problem, state, objective_name)
        return local_optima[key]

    is_stuck = is_local_optimum(current)
    stuck_count = int(is_stuck)

    history = [{
        "iteration": 0,
        "objective": value,
        "temperature": None,
        "delta_e": None,
        "exp_delta_over_temperature": None,
        "log_exp_delta_over_temperature": None,
        "exp_overflow": False,
        "acceptance_probability": None,
        "accepted": None,
        "candidate_found": True,
        "stuck": is_stuck,
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
            history.append({
                "iteration": iterations,
                "objective": value,
                "temperature": temperature,
                "delta_e": None,
                "exp_delta_over_temperature": None,
                "log_exp_delta_over_temperature": None,
                "exp_overflow": False,
                "acceptance_probability": None,
                "accepted": False,
                "candidate_found": False,
                "stuck": is_stuck,
            })
            continue

        candidate_value = score(problem, candidate)
        delta_e = candidate_value - value
        iterations += 1

        exp_delta_over_temperature, log_exp, exp_overflow = _exponential_metrics(delta_e, temperature)
        acceptance_probability = 1.0 if delta_e >= 0 else exp_delta_over_temperature
        accepted = delta_e >= 0 or rng.random() < acceptance_probability

        if accepted:
            current = candidate
            value = candidate_value

        was_stuck = is_stuck
        is_stuck = is_local_optimum(current)
        if is_stuck and not was_stuck:
            stuck_count += 1

        history.append({
            "iteration": iterations,
            "objective": value,
            "temperature": temperature,
            "delta_e": delta_e,
            "exp_delta_over_temperature": exp_delta_over_temperature,
            "log_exp_delta_over_temperature": log_exp,
            "exp_overflow": exp_overflow,
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
            "stuck_definition": "Entry episodes into states with no strictly improving legal neighbor, including the initial state",
        },
    )
