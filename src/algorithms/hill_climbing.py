

from time import perf_counter

from src.core import Problem, SearchResult, State, is_valid, neighbors, objective, random_state
from src.core.models import integer


def solve(problem: Problem, config: dict | None = None, seed: int | None = None) -> SearchResult:
    config = dict(config or {})
    unknown = set(config) - {"initial_state", "max_iterations"}
    if unknown:
        raise ValueError(f"Unknown HC parameters: {sorted(unknown)}")
    limit = config.get("max_iterations")
    if limit is not None:
        integer(limit, "max_iterations", 0)
    started = perf_counter()
    initial = config.get("initial_state")
    if initial is None:
        initial = random_state(problem, seed)
    if not isinstance(initial, State) or not is_valid(problem, initial):
        raise ValueError("initial_state must be a valid State")
    initial = initial.copy()
    current = initial.copy()
    value = objective(problem, current)
    history = [{"iteration": 0, "objective": value, "accepted": False, "neighbors_evaluated": 0}]
    scans = 0
    accepted_moves = 0
    reason = "max_iterations"
    while limit is None or scans < limit:
        best, best_value = current, value
        evaluated = 0
        for candidate in neighbors(problem, current):
            evaluated += 1
            candidate_value = objective(problem, candidate)
            if candidate_value > best_value:
                best, best_value = candidate, candidate_value
        scans += 1
        accepted = best_value > value
        history.append({"iteration": scans, "objective": best_value, "accepted": accepted,
                        "neighbors_evaluated": evaluated})
        if not accepted:
            reason = "no_improving_neighbor"
            break
        current, value = best, best_value
        accepted_moves += 1
    return SearchResult("hc", {"variant": "steepest_ascent", "max_iterations": limit}, seed,
                        initial, current.copy(), value, perf_counter() - started, scans, history,
                        stop_reason=reason, metadata={"accepted_moves": accepted_moves})
