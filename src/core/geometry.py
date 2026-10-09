from itertools import combinations

from .models import Package, Placement, Problem, State


def oriented_dimensions(package: Package, placement: Placement) -> tuple[int, int, int]:
    sizes = dict(zip(placement.orientation, (package.dimensions.w, package.dimensions.l, package.dimensions.h)))
    return sizes["x"], sizes["y"], sizes["z"]


def bounds(package: Package, placement: Placement) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    sizes = oriented_dimensions(package, placement)
    return placement.position, tuple(a + b for a, b in zip(placement.position, sizes))


def overlap(a: tuple, b: tuple, axes: tuple[int, ...] = (0, 1, 2)) -> bool:
    return all(max(a[0][i], b[0][i]) < min(a[1][i], b[1][i]) for i in axes)


def validation_errors(problem: Problem, state: State) -> list[str]:
    packages = problem.package_by_id
    if set(state.placements) != set(packages):
        return ["State must contain each package id exactly once"]
    if any(not isinstance(p, Placement) for p in state.placements.values()):
        return ["State entries must be Placement objects"]
    truck = problem.truck.dimensions
    truck_bounds = ((0, 0, 0), (truck.w, truck.l, truck.h))
    boxes = {key: bounds(packages[key], p) for key, p in state.placements.items()}
    inside = [key for key, p in state.placements.items() if p.inside]
    errors = []
    for key, p in state.placements.items():
        low, high = boxes[key]
        if p.inside:
            if not all(0 <= low[i] and high[i] <= truck_bounds[1][i] for i in range(3)):
                errors.append(f"{key}: inside package crosses truck boundary")
        elif overlap(boxes[key], truck_bounds):
            errors.append(f"{key}: outside package intersects truck interior")
    for key, other in combinations(packages, 2):
        if overlap(boxes[key], boxes[other]):
            errors.append(f"{key}/{other}: packages overlap")
    if sum(packages[key].weight for key in inside) > problem.truck.max_capacity:
        errors.append("Truck weight capacity exceeded")
    for key in inside:
        if boxes[key][0][2] == 0:
            continue
        supports = [other for other in inside if other != key
                    and boxes[other][1][2] == boxes[key][0][2]
                    and overlap(boxes[key], boxes[other], (0, 1))]
        if not supports:
            errors.append(f"{key}: no base point has support")
        for other in supports:
            if packages[other].is_fragile:
                errors.append(f"{other}: fragile package supports {key}")
    return errors


def is_valid(problem: Problem, state: State) -> bool:
    return not validation_errors(problem, state)


def objective(problem: Problem, state: State) -> float:
    errors = validation_errors(problem, state)
    if errors:
        raise ValueError("Invalid state: " + "; ".join(errors))
    return sum(p.value for p in problem.packages if state.placements[p.id].inside)

def coordinate_map(problem: Problem, state: State) -> list:
    if not is_valid(problem, state):
        raise ValueError("Cannot map an invalid state")
    d = problem.truck.dimensions
    cells = [[[None for _ in range(d.w)] for _ in range(d.l)] for _ in range(d.h)]
    for p in problem.packages:
        placement = state.placements[p.id]
        if placement.inside:
            low, high = bounds(p, placement)
            for z in range(low[2], high[2]):
                for y in range(low[1], high[1]):
                    for x in range(low[0], high[0]):
                        cells[z][y][x] = p.id
    return cells
