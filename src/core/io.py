import json
from pathlib import Path

from .geometry import coordinate_map
from .models import Dimensions, Package, Placement, Problem, SearchResult, State, Truck


def problem_from_dict(data: dict) -> Problem:
    truck_data = data["truck"]
    truck = Truck(Dimensions(**truck_data["dimensions"]), truck_data["max_capacity"])
    packages = []
    for package in data["packages"]:
        packages.append(Package(
            id=package["id"], dimensions=Dimensions(**package["dimensions"]),
            value=package["value"], weight=package["weight"],
            is_fragile=package["is_fragile"], eta=package["eta"],
        ))
    return Problem(truck, tuple(packages))


def problem_to_dict(problem: Problem) -> dict:
    return {"truck": {"dimensions": problem.truck.dimensions.to_dict(),
                      "max_capacity": problem.truck.max_capacity},
            "packages": [{"id": p.id, "dimensions": p.dimensions.to_dict(), "value": p.value,
                          "weight": p.weight, "is_fragile": p.is_fragile, "eta": p.eta}
                         for p in problem.packages]}


def state_from_dict(data: dict) -> State:
    if not isinstance(data, dict) or not isinstance(data.get("placements"), dict):
        raise ValueError("State must contain a placements object")
    return State({key: Placement(tuple(p["position"]), tuple(p["orientation"]), p["inside"])
                  for key, p in data["placements"].items()})


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: str | Path, data) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_problem(path: str | Path) -> Problem:
    return problem_from_dict(read_json(path))


def load_state(path: str | Path) -> State:
    data = read_json(path)
    if "initial_state" in data:
        data = data["initial_state"]
    return state_from_dict(data)


def save_problem(path: str | Path, problem: Problem) -> None:
    write_json(path, problem_to_dict(problem))


def save_state(path: str | Path, state: State) -> None:
    write_json(path, state.to_dict())


def result_from_dict(data: dict) -> SearchResult:
    return SearchResult(algorithm=data["algorithm"], parameters=data["parameters"], seed=data["seed"],
                        initial_state=state_from_dict(data["initial_state"]),
                        final_state=state_from_dict(data["final_state"]),
                        final_objective=data["final_objective"], duration_seconds=data["duration_seconds"],
                        iterations=data["iterations"], history=data["history"],
                        initial_population=[state_from_dict(s) for s in data.get("initial_population", [])],
                        stop_reason=data.get("stop_reason", ""), is_stub=data.get("is_stub", False),
                        metadata=data.get("metadata", {}))


def load_result(path: str | Path) -> SearchResult:
    return result_from_dict(read_json(path))


def save_result(path: str | Path, problem: Problem, result: SearchResult) -> None:
    data = result.to_dict()
    data["problem"] = problem_to_dict(problem)
    data["initial_coordinate_map"] = coordinate_map(problem, result.initial_state)
    data["final_coordinate_map"] = coordinate_map(problem, result.final_state)
    write_json(path, data)
