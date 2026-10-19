

from random import Random
from time import perf_counter

from src.core import Problem, SearchResult, State, get_objective, is_valid, random_state, random_successor
from src.core.models import integer

from . import hill_climbing


def _validate_config(config: dict | None, extra: set[str]) -> dict:
    config = dict(config or {})
    unknown = set(config) - {"initial_state", "max_iterations", "objective"} - extra
    if unknown:
        raise ValueError(f"Unknown HC parameters: {sorted(unknown)}")
    if config.get("max_iterations") is not None:
        integer(config["max_iterations"], "max_iterations", 0)
    config.setdefault("objective", "value")
    get_objective(config["objective"])
    return config


def solve_sideways(problem: Problem, config: dict | None = None,
                   seed: int | None = None) -> SearchResult:
    config = _validate_config(config, {"max_sideways_moves"})
    config.setdefault("max_sideways_moves", 100)
    integer(config["max_sideways_moves"], "max_sideways_moves", 0)
    return _climb(problem, config, seed)


def solve_stochastic(problem: Problem, config: dict | None = None,
                     seed: int | None = None) -> SearchResult:
    config = _validate_config(config, set())
    limit = config.get("max_iterations")
    if limit is None:
        limit = 1000
    score = get_objective(config["objective"])
    started = perf_counter()
    rng = Random(seed)
    initial = config.get("initial_state")
    if initial is None:
        initial = random_state(problem, seed)
    if not isinstance(initial, State) or not is_valid(problem, initial):
        raise ValueError("initial_state must be a valid State")
    initial = initial.copy()
    current = initial.copy()
    value = score(problem, current)
    history = [{"iteration": 0, "objective": value, "accepted": False, "neighbors_evaluated": 0}]
    iterations = accepted_moves = 0
    reason = "max_iterations"
    for iterations in range(1, limit + 1):
        candidate = random_successor(problem, current, rng)
        if candidate is None:
            history.append({"iteration": iterations, "objective": value, "accepted": False,
                            "neighbors_evaluated": 0})
            reason = "no_valid_neighbor"
            break
        candidate_value = score(problem, candidate)
        accepted = candidate_value > value
        if accepted:
            current, value = candidate, candidate_value
            accepted_moves += 1
        history.append({"iteration": iterations, "objective": value, "accepted": accepted,
                        "neighbors_evaluated": 1})
    return SearchResult("hc", {"variant": "stochastic", "max_iterations": limit,
                               "objective": config["objective"]}, seed,
                        initial, current.copy(), value, perf_counter() - started, iterations, history,
                        stop_reason=reason, metadata={"accepted_moves": accepted_moves})


def solve_random_restart(problem: Problem, config: dict | None = None,
                         seed: int | None = None) -> SearchResult:
    config = _validate_config(config, {"max_restarts"})
    max_restarts = config.get("max_restarts", 10)
    integer(max_restarts, "max_restarts", 0)
    limit = config.get("max_iterations")
    started = perf_counter()
    rng = Random(seed)
    history, runs = [], []
    iterations = accepted_moves = best_restart = 0
    best_value = None
    best_result = None
    initial = None
    for restart in range(max_restarts + 1):
        run_seed = seed if restart == 0 else rng.getrandbits(64)
        run_config = {"max_iterations": limit, "objective": config["objective"]}
        if restart == 0:
            run_config["initial_state"] = config.get("initial_state")
        else:
            run_config["initial_state"] = random_state(problem, run_seed)
        result = hill_climbing.solve(problem, run_config, run_seed)
        if restart == 0:
            initial = result.initial_state.copy()
        for entry in result.history:
            best_value = entry["objective"] if best_value is None else max(best_value, entry["objective"])
            row = {**entry, "iteration": iterations + entry["iteration"],
                   "restart": restart, "run_iteration": entry["iteration"],
                   "best_objective": best_value}
            if entry["iteration"] == 0:
                row["event"] = "initial" if restart == 0 else "restart"
            history.append(row)
        if best_result is None or result.final_objective > best_result.final_objective:
            best_result, best_restart = result, restart
        iterations += result.iterations
        accepted_moves += result.metadata["accepted_moves"]
        runs.append({"restart": restart, "seed": run_seed, "iterations": result.iterations,
                     "initial_objective": result.history[0]["objective"],
                     "final_objective": result.final_objective,
                     "initial_state": result.initial_state.to_dict(),
                     "final_state": result.final_state.to_dict(),
                     "duration_seconds": result.duration_seconds, "stop_reason": result.stop_reason})

    return SearchResult("hc", {"variant": "random_restart", "max_iterations": limit,
                               "max_restarts": max_restarts, "objective": config["objective"]}, seed, initial,
                        best_result.final_state.copy(), best_result.final_objective,
                        perf_counter() - started, iterations, history, stop_reason="max_restarts",
                        metadata={"accepted_moves": accepted_moves, "restart_count": max_restarts,
                                  "iterations_per_restart": [run["iterations"] for run in runs],
                                  "best_restart": best_restart, "runs": runs})


def _climb(problem: Problem, config: dict, seed: int | None) -> SearchResult:
    score = get_objective(config["objective"])
    started = perf_counter()
    rng = Random(seed)
    initial = config.get("initial_state")
    if initial is None:
        initial = random_state(problem, seed)
    if not isinstance(initial, State) or not is_valid(problem, initial):
        raise ValueError("initial_state must be a valid State")
    initial = initial.copy()
    current = initial.copy()
    value = score(problem, current)
    limit = config.get("max_iterations")
    sideways_limit = config.get("max_sideways_moves", 0)
    scans = accepted_moves = sideways_moves = consecutive_sideways_moves = 0
    reason = "max_iterations"
    history = []

    def record(accepted: bool, evaluated: int, is_sideways: bool = False) -> None:
        entry = {"iteration": scans, "objective": value, "accepted": accepted,
                 "neighbors_evaluated": evaluated}
        entry.update(sideways=is_sideways, sideways_moves=sideways_moves,
                     consecutive_sideways_moves=consecutive_sideways_moves)
        history.append(entry)

    record(False, 0)
    while limit is None or scans < limit:
        chosen, chosen_value, evaluated = hill_climbing._best_neighbor(
            problem, current, value, rng, allow_equal=True, score=score)
        scans += 1
        accepted = chosen is not None and (chosen_value > value or consecutive_sideways_moves < sideways_limit)
        is_sideways = accepted and chosen_value == value
        if accepted:
            current, value = chosen, chosen_value
            accepted_moves += 1
            sideways_moves += int(is_sideways)
            consecutive_sideways_moves = consecutive_sideways_moves + 1 if is_sideways else 0
        record(accepted, evaluated, is_sideways)
        if not accepted:
            reason = "max_sideways_moves" if chosen is not None else "no_improving_neighbor"
            break
        if is_sideways and consecutive_sideways_moves == sideways_limit:
            reason = "max_sideways_moves"
            break

    parameters = {"variant": "sideways", "max_iterations": limit, "max_sideways_moves": sideways_limit,
                  "objective": config["objective"]}
    metadata = {"accepted_moves": accepted_moves, "sideways_moves": sideways_moves,
                "consecutive_sideways_moves": consecutive_sideways_moves}
    return SearchResult("hc", parameters, seed, initial, current.copy(), value,
                        perf_counter() - started, scans, history,
                        stop_reason=reason, metadata=metadata)
