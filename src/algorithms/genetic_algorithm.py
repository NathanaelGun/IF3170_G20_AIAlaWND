from dataclasses import replace
from random import Random
from time import perf_counter

from src.core import Problem, SearchResult, State, empty_state, is_valid, objective, random_neighbor, random_state
from src.core.models import integer

CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
ELITISM = 1


def _roulette_select(population: list[State], fitnesses: list[float], total_fitness: float, rng: Random) -> State:
    if total_fitness <= 0:
        return rng.choice(population)
    pick = rng.uniform(0.0, total_fitness)
    current = 0.0
    for individual, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return individual
    return population[-1]


def _repair_child(problem: Problem, raw_placements: dict) -> State:
    child = empty_state(problem)
    # Ambil gen inside == True, urutkan berdasarkan z menaik (bawah ke atas)
    inside_candidates = [
        p for p in problem.packages if raw_placements[p.id].inside
    ]
    inside_candidates.sort(key=lambda p: raw_placements[p.id].position[2])

    for pkg in inside_candidates:
        candidate = child.copy()
        candidate.placements[pkg.id] = raw_placements[pkg.id]
        if is_valid(problem, candidate):
            child = candidate

    # Gen outside tetap di luar, mempertahankan orientasi yang diwarisi
    for pkg in problem.packages:
        if not child.placements[pkg.id].inside:
            orientation = raw_placements[pkg.id].orientation
            child.placements[pkg.id] = replace(child.placements[pkg.id], orientation=orientation)
    return child


def _crossover(problem: Problem, parent_a: State, parent_b: State, rng: Random) -> tuple[State, State]:
    packages = problem.packages
    n = len(packages)
    if n < 2 or rng.random() >= CROSSOVER_RATE:
        return parent_a.copy(), parent_b.copy()

    k = rng.randint(1, n - 1)
    raw1 = {}
    raw2 = {}
    for i, p in enumerate(packages):
        if i < k:
            raw1[p.id] = parent_a.placements[p.id]
            raw2[p.id] = parent_b.placements[p.id]
        else:
            raw1[p.id] = parent_b.placements[p.id]
            raw2[p.id] = parent_a.placements[p.id]

    child1 = _repair_child(problem, raw1)
    child2 = _repair_child(problem, raw2)
    return child1, child2


def _mutate(problem: Problem, individual: State, rng: Random) -> State:
    if rng.random() < MUTATION_RATE:
        mutant = random_neighbor(problem, individual, rng)
        if mutant is not None:
            return mutant
    return individual


def solve(problem: Problem, config: dict | None = None, seed: int | None = None) -> SearchResult:
    config = dict(config or {})
    unknown = set(config) - {"population_size", "max_iterations"}
    if unknown:
        raise ValueError(f"Unknown GA parameters: {sorted(unknown)}")

    population_size = config.get("population_size", 20)
    max_iterations = config.get("max_iterations", 50)

    integer(population_size, "population_size", 2)
    integer(max_iterations, "max_iterations", 1)

    rng = Random(seed)
    started = perf_counter()

    # Inisialisasi populasi awal (generasi 0)
    population = [random_state(problem, rng.randint(0, 10**9)) for _ in range(population_size)]
    initial_population = [s.copy() for s in population]

    fitnesses = [objective(problem, s) for s in population]
    max_fit = max(fitnesses)
    mean_fit = sum(fitnesses) / len(fitnesses)

    best_individual = population[fitnesses.index(max_fit)].copy()
    initial_best = best_individual.copy()

    history = [{
        "iteration": 0,
        "objective": max_fit,
        "max_objective": max_fit,
        "mean_objective": mean_fit,
    }]

    for gen in range(1, max_iterations + 1):
        total_fit = sum(fitnesses)
        new_population = [best_individual.copy()]  # Elitisme

        while len(new_population) < population_size:
            p1 = _roulette_select(population, fitnesses, total_fit, rng)
            p2 = _roulette_select(population, fitnesses, total_fit, rng)

            c1, c2 = _crossover(problem, p1, p2, rng)
            c1 = _mutate(problem, c1, rng)
            c2 = _mutate(problem, c2, rng)

            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population
        fitnesses = [objective(problem, s) for s in population]
        gen_max = max(fitnesses)
        gen_mean = sum(fitnesses) / len(fitnesses)

        if gen_max > objective(problem, best_individual):
            best_individual = population[fitnesses.index(gen_max)].copy()

        history.append({
            "iteration": gen,
            "objective": max(gen_max, objective(problem, best_individual)),
            "max_objective": max(gen_max, objective(problem, best_individual)),
            "mean_objective": gen_mean,
        })

    duration = perf_counter() - started
    final_obj = objective(problem, best_individual)

    return SearchResult(
        algorithm="ga",
        parameters={
            "population_size": population_size,
            "max_iterations": max_iterations,
        },
        seed=seed,
        initial_state=initial_best,
        final_state=best_individual.copy(),
        final_objective=final_obj,
        duration_seconds=duration,
        iterations=max_iterations,
        history=history,
        initial_population=initial_population,
        stop_reason="max_iterations",
        metadata={
            "elitism": ELITISM,
            "crossover_rate": CROSSOVER_RATE,
            "mutation_rate": MUTATION_RATE,
        },
    )
