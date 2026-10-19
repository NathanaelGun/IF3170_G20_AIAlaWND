from dataclasses import dataclass, field
from itertools import permutations
from math import isfinite
from typing import Any

ORIENTATIONS = tuple(permutations(("x", "y", "z")))


def integer(value: Any, name: str, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))


def number(value: Any, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class Dimensions:
    w: int
    l: int
    h: int

    def __post_init__(self) -> None:
        for name in ("w", "l", "h"):
            integer(getattr(self, name), name, 1)

    def to_dict(self) -> dict:
        return {"w": self.w, "l": self.l, "h": self.h}


@dataclass(frozen=True)
class Truck:
    dimensions: Dimensions
    max_capacity: float

    def __post_init__(self) -> None:
        number(self.max_capacity, "max_capacity")


@dataclass(frozen=True)
class Package:
    id: str
    dimensions: Dimensions
    value: float
    weight: float
    is_fragile: bool
    eta: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Package id must be a nonempty string")
        number(self.value, "value")
        number(self.weight, "weight")
        if type(self.is_fragile) is not bool:
            raise ValueError("is_fragile must be a boolean")
        integer(self.eta, "eta", 0)


@dataclass(frozen=True)
class Problem:
    trucks: tuple[Truck, ...] # dana
    packages: tuple[Package, ...]

    def __post_init__(self) -> None:
        if len({p.id for p in self.packages}) != len(self.packages):
            raise ValueError("Package ids must be unique")
        if not self.trucks:
            raise ValueError("Problem must contain at least one truck")

    @property
    def truck(self) -> Truck:
        return self.trucks[0]

    @property
    def package_by_id(self) -> dict[str, Package]:
        return {p.id: p for p in self.packages}


@dataclass(frozen=True)
class Placement:
    position: tuple[int, int, int]
    orientation: tuple[str, str, str] = ("x", "y", "z")
    truck_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, tuple) or len(self.position) != 3:
            raise ValueError("position must be a tuple of three integers")
        for value in self.position:
            integer(value, "coordinate")
        if self.orientation not in ORIENTATIONS:
            raise ValueError("orientation must map w/l/h to a permutation of x/y/z")
        if self.truck_index is not None:                            # dana
            integer(self.truck_index, "truck_index", 0)

    @property
    def inside(self) -> bool:
        return self.truck_index is not None

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "truck_index": self.truck_index,
            "inside": self.inside,
        }


@dataclass
class State:
    placements: dict[str, Placement]

    def copy(self) -> "State":
        return State(self.placements.copy())

    def to_dict(self) -> dict:
        return {"placements": {key: value.to_dict() for key, value in self.placements.items()}}


@dataclass
class SearchResult:
    algorithm: str
    parameters: dict[str, Any]
    seed: int | None
    initial_state: State
    final_state: State
    final_objective: float
    duration_seconds: float
    iterations: int
    history: list[dict[str, Any]]
    initial_population: list[State] = field(default_factory=list)
    stop_reason: str = ""
    is_stub: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "seed": self.seed,
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "final_objective": self.final_objective,
            "duration_seconds": self.duration_seconds,
            "iterations": self.iterations,
            "history": self.history,
            "initial_population": [s.to_dict() for s in self.initial_population],
            "stop_reason": self.stop_reason,
            "is_stub": self.is_stub,
            "metadata": self.metadata,
        }
