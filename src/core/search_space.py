from dataclasses import replace
from itertools import combinations
from random import Random
from typing import Iterator

from .geometry import is_valid, oriented_dimensions
from .models import ORIENTATIONS, Placement, Problem, State


def outside_position(problem: Problem, index: int) -> tuple[int, int, int]:
    stride = max((max(p.dimensions.w, p.dimensions.l, p.dimensions.h)
                  for p in problem.packages), default=1) + 1
    return problem.truck.dimensions.w + 1 + index * stride, 0, 0


def empty_state(problem: Problem) -> State:
    return State({p.id: Placement(outside_position(problem, i)) for i, p in enumerate(problem.packages)})


def move_package(state: State, package_id: str, position: tuple[int, int, int], inside: bool) -> State:
    candidate = state.copy()
    candidate.placements[package_id] = replace(state.placements[package_id], position=position, inside=inside)
    return candidate


def swap_packages(state: State, first: str, second: str) -> State:
    candidate = state.copy()
    a, b = state.placements[first], state.placements[second]
    candidate.placements[first] = replace(a, position=b.position, inside=b.inside)
    candidate.placements[second] = replace(b, position=a.position, inside=a.inside)
    return candidate


def rotate_package(state: State, package_id: str, axis: str) -> State:
    if axis not in ("x", "y", "z"):
        raise ValueError("Rotation axis must be x, y, or z")
    first, second = [name for name in ("x", "y", "z") if name != axis]
    rotated_axes = {axis: axis, first: second, second: first}
    orientation = tuple(rotated_axes[name] for name in state.placements[package_id].orientation)
    candidate = state.copy()
    candidate.placements[package_id] = replace(state.placements[package_id], orientation=orientation)
    return candidate


def inside_positions(problem: Problem) -> Iterator[tuple[int, int, int]]:
    d = problem.truck.dimensions
    for z in range(d.h):
        for y in range(d.l):
            for x in range(d.w):
                yield x, y, z


def _try_placements(problem: Problem, state: State, package_id: str, orientations, positions) -> State:
    for orientation in orientations:
        for position in positions:
            candidate = state.copy()
            candidate.placements[package_id] = Placement(position, orientation, True)
            if is_valid(problem, candidate):
                return candidate
    return state


def random_state(problem: Problem, seed: int | None = None) -> State:
    rng = Random(seed)
    state = empty_state(problem)
    indices = list(range(len(problem.packages)))
    rng.shuffle(indices)
    for i in indices:
        package = problem.packages[i]
        placement = state.placements[package.id]
        state.placements[package.id] = replace(placement, orientation=rng.choice(ORIENTATIONS))

        if rng.random() < 0.3:
            continue
        orientations = list(ORIENTATIONS)
        rng.shuffle(orientations)
        positions = list(inside_positions(problem))
        rng.shuffle(positions)
        state = _try_placements(problem, state, package.id, orientations, positions)
    return state


def neighbors(problem: Problem, state: State) -> Iterator[State]:
    if not is_valid(problem, state):
        raise ValueError("Neighborhood requires a valid source state")
    ids = [p.id for p in problem.packages]
    parking_positions = [outside_position(problem, slot) for slot in range(len(ids))]

    def raw_candidates() -> Iterator[State]:
        for a, b in combinations(ids, 2):
            yield swap_packages(state, a, b)
        for key in ids:
            for position in inside_positions(problem):
                yield move_package(state, key, position, True)
            for position in parking_positions:
                yield move_package(state, key, position, False)
            for axis in ("x", "y", "z"):
                yield rotate_package(state, key, axis)

    def signature(s: State) -> tuple:
        return tuple(s.placements[key] for key in ids)

    seen = {signature(state)}
    for candidate in raw_candidates():
        key = signature(candidate)
        if key not in seen and is_valid(problem, candidate):
            seen.add(key)
            yield candidate


def sample_neighbor(problem: Problem, state: State, rng: Random) -> State | None:
    chosen = None
    for count, candidate in enumerate(neighbors(problem, state), 1):
        if rng.randrange(count) == 0:
            chosen = candidate
    return chosen


def random_neighbor(problem: Problem, state: State, rng: Random, tries: int = 100) -> State | None:
    if not problem.packages:
        return None
    ids = [p.id for p in problem.packages]
    n = len(ids)
    available_moves = ["move", "rotate", "swap"] if n >= 2 else ["move", "rotate"]

    for _ in range(tries):
        move_type = rng.choice(available_moves)
        if move_type == "swap":
            a, b = rng.sample(ids, 2)
            candidate = swap_packages(state, a, b)
        elif move_type == "rotate":
            pkg_id = rng.choice(ids)
            axis = rng.choice(("x", "y", "z"))
            candidate = rotate_package(state, pkg_id, axis)
        else:  # move
            pkg_id = rng.choice(ids)
            if rng.random() < 0.5:
                # Coba tempatkan di dalam truk
                inside_tops = [
                    state.placements[other.id].position[2] + oriented_dimensions(other, state.placements[other.id])[2]
                    for other in problem.packages
                    if other.id != pkg_id and state.placements[other.id].inside
                ]
                possible_z = [0] + [z for z in inside_tops if z < problem.truck.dimensions.h]
                z = rng.choice(possible_z)
                x = rng.randrange(problem.truck.dimensions.w)
                y = rng.randrange(problem.truck.dimensions.l)
                candidate = move_package(state, pkg_id, (x, y, z), True)
            else:
                # Pindahkan ke slot parkir luar
                slot = rng.randrange(n)
                candidate = move_package(state, pkg_id, outside_position(problem, slot), False)

        if candidate.placements != state.placements and is_valid(problem, candidate):
            return candidate

    return None
