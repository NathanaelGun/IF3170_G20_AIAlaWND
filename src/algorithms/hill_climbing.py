

from random import Random
from time import perf_counter

from src.core import Problem, SearchResult, State, get_objective, is_valid, neighbors, objective, random_state
from src.core.models import integer


def _best_neighbor(problem: Problem, current: State, value: float, rng: Random,
                   allow_equal: bool = False, score=objective) -> tuple[State | None, float, int]:
    best, best_value = None, value
    tied = evaluated = 0
    for candidate in neighbors(problem, current):
        evaluated += 1
        candidate_value = score(problem, candidate)
        if candidate_value > best_value:
            best, best_value, tied = candidate, candidate_value, 1
        elif candidate_value == best_value and (best is not None or allow_equal):
            tied += 1
            if rng.randrange(tied) == 0:
                best = candidate
    return best, best_value, evaluated


def solve(problem: Problem, config: dict | None = None, seed: int | None = None) -> SearchResult:
    config = dict(config or {})
    unknown = set(config) - {"initial_state", "max_iterations", "objective"}
    if unknown:
        raise ValueError(f"Unknown HC parameters: {sorted(unknown)}")
    objective_name = config.get("objective", "value")
    score = get_objective(objective_name)
    limit = config.get("max_iterations")
    if limit is not None:
        integer(limit, "max_iterations", 0)
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
    scans = 0
    accepted_moves = 0
    reason = "max_iterations"
    while limit is None or scans < limit:
        best, best_value, evaluated = _best_neighbor(problem, current, value, rng, score=score)
        scans += 1
        accepted = best_value > value
        history.append({"iteration": scans, "objective": best_value, "accepted": accepted,
                        "neighbors_evaluated": evaluated})
        if not accepted:
            reason = "no_improving_neighbor"
            break
        current, value = best, best_value
        accepted_moves += 1
    return SearchResult("hc", {"variant": "steepest_ascent", "max_iterations": limit, "objective": objective_name},
                        seed, initial, current.copy(), value, perf_counter() - started, scans, history,
                        stop_reason=reason, metadata={"accepted_moves": accepted_moves})
