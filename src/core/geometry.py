from itertools import combinations

from .models import Package, Placement, Problem, State


def oriented_dimensions(package: Package, placement: Placement) -> tuple[int, int, int]:
    sizes = dict(zip(placement.orientation, (package.dimensions.w, package.dimensions.l, package.dimensions.h),))
    return sizes["x"], sizes["y"], sizes["z"]


def bounds(package: Package, placement: Placement) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    sizes = oriented_dimensions(package, placement)
    return placement.position, tuple(a + b for a, b in zip(placement.position, sizes))


def overlap(a: tuple, b: tuple, axes: tuple[int, ...] = (0, 1, 2)) -> bool:
    return all(max(a[0][i], b[0][i]) < min(a[1][i], b[1][i]) for i in axes)

#dana
def validation_errors(problem: Problem, state: State) -> list[str]:
    packages = problem.package_by_id
    if set(state.placements) != set(packages):
        return ["State must contain each package id exactly once"]
    if any(not isinstance(p, Placement) for p in state.placements.values()):
        return ["State entries must be Placement objects"]
    errors = []
    for key, placement in state.placements.items():
        index = placement.truck_index
        if index is not None:
            if type(index) is not int or not 0 <= index < len(problem.trucks):
                errors.append(f"{key}: invalid truck index")
    if errors:
        return errors
    boxes = {key: bounds(packages[key], p) for key, p in state.placements.items()}
    for key, p in state.placements.items():
        low, high = boxes[key]
        if p.inside:
            truck = problem.trucks[p.truck_index].dimensions
            truck_bounds = ((0, 0, 0), (truck.w, truck.l, truck.h))
            if not all(0 <= low[i] and high[i] <= truck_bounds[1][i] for i in range(3)):
                errors.append(f"{key}: inside package crosses truck boundary")
    for key, other in combinations(packages, 2):
        idx_a, idx_b = state.placements[key].truck_index, state.placements[other].truck_index
        if idx_a is not None and idx_a == idx_b and overlap(boxes[key], boxes[other]):
            errors.append(f"{key}/{other}: packages overlap")
    for index, truck in enumerate(problem.trucks):
        inside = [key for key, p in state.placements.items() if p.truck_index == index]
        if sum(packages[key].weight for key in inside) > truck.max_capacity:
            errors.append(f"Truck {index}: weight capacity exceeded")
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


def objective_eta_priority(problem: Problem, state: State) -> float:
    """Objective alternatif: value paket di dalam truk dibobot urgensi ETA, yaitu sum(value / (1 + eta)).

    Paket dengan ETA kecil (harus segera sampai) bernilai lebih tinggi. Hanya memakai atribut yang sudah ada.
    """
    errors = validation_errors(problem, state)
    if errors:
        raise ValueError("Invalid state: " + "; ".join(errors))
    return round(sum(p.value / (1 + p.eta) for p in problem.packages if state.placements[p.id].inside), 6)


OBJECTIVES = {"value": objective, "eta_priority": objective_eta_priority}


def get_objective(name: str = "value"):
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {name}. Pilihan: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name]

def coordinate_map(problem: Problem, state: State) -> list:
    if not is_valid(problem, state):
        raise ValueError("Cannot map an invalid state")
    truck_cells = []
    for truck in problem.trucks:
        d = truck.dimensions
        cells = [[[None for _ in range(d.w)] for _ in range(d.l)] for _ in range(d.h)]
        truck_cells.append(cells)
    for p in problem.packages:
        placement = state.placements[p.id]
        if placement.inside:
            cells = truck_cells[placement.truck_index]
            low, high = bounds(p, placement)
            for z in range(low[2], high[2]):
                for y in range(low[1], high[1]):
                    for x in range(low[0], high[0]):
                        cells[z][y][x] = p.id
    return truck_cells
