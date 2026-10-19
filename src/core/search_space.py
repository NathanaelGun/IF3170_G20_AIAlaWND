from dataclasses import replace
from itertools import combinations
from random import Random
from typing import Iterator

from .geometry import is_valid, oriented_dimensions
from .models import ORIENTATIONS, Placement, Problem, State

# dana, posisi parkir tidak bergantung ke lebar truk
def outside_position(problem: Problem, index: int) -> tuple[int, int, int]:
    stride = max((max(p.dimensions.w, p.dimensions.l, p.dimensions.h)
                  for p in problem.packages), default=1) + 1
    return index * stride, 0, 0


def empty_state(problem: Problem) -> State:
    return State({p.id: Placement(outside_position(problem, i)) for i, p in enumerate(problem.packages)})

# dana, di move_package dan swap_packages ganti inside dengan truck_index
def move_package(state: State, package_id: str, position: tuple[int, int, int], truck_index: int | None) -> State:
    candidate = state.copy()
    candidate.placements[package_id] = replace(state.placements[package_id], position=position, truck_index=truck_index)
    return candidate


def swap_packages(state: State, first: str, second: str) -> State:
    candidate = state.copy()
    a, b = state.placements[first], state.placements[second]
    candidate.placements[first] = replace(a, position=b.position, truck_index=b.truck_index)
    candidate.placements[second] = replace(b, position=a.position, truck_index=a.truck_index)
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

#dana, bisa untuk beda truk
def inside_positions(problem: Problem, truck_index: int) -> Iterator[tuple[int, int, int]]:
    d = problem.trucks[truck_index].dimensions
    for z in range(d.h):
        for y in range(d.l):
            for x in range(d.w):
                yield x, y, z

# dana, pake placement dengan indeks truk
def _try_placements(problem: Problem, state: State, package_id: str, orientations, positions, truck_index: int) -> State:
    for orientation in orientations:
        for position in positions:
            candidate = state.copy()
            candidate.placements[package_id] = Placement(position, orientation, truck_index=truck_index)
            if is_valid(problem, candidate):
                return candidate
    return state

# dana, random urutan truk yg dicoba juga
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
        truck_indices = list(range(len(problem.trucks)))
        rng.shuffle(truck_indices)
        for truck_index in truck_indices:
            positions = list(inside_positions(problem, truck_index))
            rng.shuffle(positions)
            state = _try_placements(problem, state, package.id, orientations, positions, truck_index)
            if state.placements[package.id].inside:
                break
    return state

# dana, perpindahan dicoba ke setiap truk, dan perpindahan keluar memakai None
def neighbors(problem: Problem, state: State) -> Iterator[State]:
    if not is_valid(problem, state):
        raise ValueError("Neighborhood requires a valid source state")
    ids = [p.id for p in problem.packages]
    parking_positions = [outside_position(problem, slot) for slot in range(len(ids))]

    def raw_candidates() -> Iterator[State]:
        for a, b in combinations(ids, 2):
            yield swap_packages(state, a, b)
        for key in ids:
            for truck_index in range(len(problem.trucks)):
                for position in inside_positions(problem, truck_index):
                    yield move_package(state, key, position, truck_index)
            for position in parking_positions:
                yield move_package(state, key, position, None)
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

# dana, truk tujuan acak, koordinat ngikut dimensi truknya
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
                truck_index = rng.randrange(len(problem.trucks))
                d = problem.trucks[truck_index].dimensions
                # Coba tempatkan di dalam truk
                inside_tops = [
                    state.placements[other.id].position[2] + oriented_dimensions(other, state.placements[other.id])[2]
                    for other in problem.packages
                    if other.id != pkg_id and state.placements[other.id].truck_index == truck_index
                ]
                possible_z = [0] + [z for z in inside_tops if z < d.h]
                z = rng.choice(possible_z)
                x = rng.randrange(d.w)
                y = rng.randrange(d.l)
                candidate = move_package(state, pkg_id, (x, y, z), truck_index)
            else:
                # Pindahkan ke slot parkir luar
                slot = rng.randrange(n)
                candidate = move_package(state, pkg_id, outside_position(problem, slot), None)

        if candidate.placements != state.placements and is_valid(problem, candidate):
            return candidate

    return None


def random_successor(problem: Problem, state: State, rng: Random) -> State | None:
    
    if not is_valid(problem, state):
        raise ValueError("Random successor requires a valid source state")
    ids = [package.id for package in problem.packages]
    n = len(ids)
    d = problem.truck.dimensions
    volume = d.w * d.l * d.h
    swap_count = n * (n - 1) // 2
    move_count = n * (volume + n)
    remaining = swap_count + move_count + 3 * n
    shuffled = {}
    while remaining:
    
        slot = rng.randrange(remaining)
        action = shuffled.get(slot, slot)
        remaining -= 1
        shuffled[slot] = shuffled.get(remaining, remaining)
        if action < swap_count:
            first = 0
            while action >= n - first - 1:
                action -= n - first - 1
                first += 1
            candidate = swap_packages(state, ids[first], ids[first + 1 + action])
        elif action < swap_count + move_count:
            package_index, position_index = divmod(action - swap_count, volume + n)
            if position_index < volume:
                z, floor_index = divmod(position_index, d.w * d.l)
                y, x = divmod(floor_index, d.w)
                candidate = move_package(state, ids[package_index], (x, y, z), True)
            else:
                position = outside_position(problem, position_index - volume)
                candidate = move_package(state, ids[package_index], position, False)
        else:
            package_index, axis_index = divmod(action - swap_count - move_count, 3)
            candidate = rotate_package(state, ids[package_index], ("x", "y", "z")[axis_index])
        if candidate != state and is_valid(problem, candidate):
            return candidate
    return None
