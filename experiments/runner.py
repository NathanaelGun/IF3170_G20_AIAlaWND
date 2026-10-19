import csv
from math import isfinite, log
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms import genetic_algorithm, hill_climbing, hill_climbing_variants, simulated_annealing
from src.core import Problem, SearchResult, get_objective
from src.core.io import save_result
from src.core.models import integer


def _validate_seeds(seeds):
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Eksperimen memerlukan tepat 3 seed berbeda")
    for seed in seeds:
        integer(seed, "seed")


def _validate_variants(values, label, minimum):
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError(f"{label} memerlukan tepat 3 variasi berbeda")
    for value in values:
        integer(value, label, minimum)


def plot_objective(result: SearchResult, path: Path | str) -> None:
    iterations = [h["iteration"] for h in result.history]
    objectives = [h["objective"] for h in result.history]

    plt.figure(figsize=(8, 5))
    marker = "o" if len(iterations) <= 50 else None
    plt.plot(iterations, objectives, marker=marker, color="#1f77b4", label="Objective")
    plt.title(f"{result.algorithm.upper()} - Objective vs Iterasi (Seed {result.seed})")
    plt.xlabel("Banyak Iterasi")
    plt.ylabel("Nilai Objective Function")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sa_probability(result: SearchResult, path: Path | str) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # Plot the exponent in logarithmic coordinates, with ticks displaying exp(Delta/T).
    # This preserves values above one and float-overflowing exponentials.
    points = []
    omitted = 0
    for entry in result.history:
        exponent = entry.get("log_exp_delta_over_temperature")
        if exponent is None:
            raw = entry.get("exp_delta_over_temperature")
            exponent = log(raw) if raw is not None and raw > 0 else None
        if entry["iteration"] > 0 and exponent is not None:
            value = float(exponent) / log(10)
            if isfinite(value):
                points.append((entry["iteration"], value))
            else:
                omitted += 1
    figure, axis = plt.subplots(figsize=(8, 5))
    if points:
        iterations, values = zip(*points)
        axis.plot(iterations, values, marker="o" if len(points) <= 50 else None,
                  color="#d62728", alpha=0.7, label=r"$e^{\Delta E / T}$")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Tidak ada kandidat untuk menghitung exp(Delta E / T)",
                  transform=axis.transAxes, ha="center")
    if omitted:
        axis.text(0.02, 0.02, f"{omitted} eksponen melampaui jangkauan koordinat float; lihat JSON.",
                  transform=axis.transAxes, fontsize=8)
    axis.set(title=rf"SA - $e^{{\Delta E / T}}$ vs Iterasi (Seed {result.seed})",
             xlabel="Banyak Iterasi", ylabel=r"$e^{\Delta E / T}$ (skala logaritmik)")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"$10^{{{value:g}}}$"))
    axis.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axis.grid(True, linestyle="--", alpha=0.4)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_ga_progress(result: SearchResult, path: Path | str) -> None:
    iterations = [h["iteration"] for h in result.history]
    max_objs = [h.get("max_objective", h["objective"]) for h in result.history]
    mean_objs = [h.get("mean_objective", h["objective"]) for h in result.history]

    plt.figure(figsize=(8, 5))
    marker = "o" if len(iterations) <= 50 else None
    plt.plot(iterations, max_objs, marker=marker, color="#1f77b4", label="Maksimum Populasi")
    plt.plot(iterations, mean_objs, marker=marker, color="#ff7f0e", linestyle="--", label="Rata-rata Populasi")
    plt.title(f"Genetic Algorithm - Objective vs Iterasi (Seed {result.seed})")
    plt.xlabel("Banyak Iterasi (Generasi)")
    plt.ylabel("Nilai Objective Function")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150, bbox_inches="tight")
    plt.close()


def _write_summary_csv(output_dir: Path, records: list[dict]) -> None:
    if not records:
        return
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "algorithm",
        "objective",
        "experiment",
        "population",
        "iterations",
        "seed",
        "initial_objective",
        "final_objective",
        "duration_seconds",
        "stop_reason",
        "stuck_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Rekapitulasi CSV tersimpan di: {csv_path.resolve()}")


def run_hc_experiments(
    problem: Problem,
    output_dir: Path | str,
    seeds: Sequence[int] = (42, 100, 2026),
    max_iterations: int | None = None,
    objective_name: str = "value",
    variant: str = "steepest_ascent",
    max_sideways_moves: int = 20,
    max_restarts: int = 3,
    run_configs: Sequence[dict] | None = None,
) -> int:
    _validate_seeds(seeds)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    objective = get_objective(objective_name)
    config = {"objective": objective_name}
    if max_iterations is not None:
        config["max_iterations"] = max_iterations
    solvers = {"steepest_ascent": hill_climbing.solve,
               "sideways": hill_climbing_variants.solve_sideways,
               "random_restart": hill_climbing_variants.solve_random_restart,
               "stochastic": hill_climbing_variants.solve_stochastic}
    if variant == "sideways":
        config["max_sideways_moves"] = max_sideways_moves
    elif variant == "random_restart":
        config["max_restarts"] = max_restarts
    if run_configs is None:
        run_configs = ({"variant": variant, **config},) * len(seeds)
    if len(run_configs) != len(seeds):
        raise ValueError("Setiap run eksperimen HC harus memiliki konfigurasi")
    prepared = []
    for settings in run_configs:
        run_config = dict(settings)
        run_variant = run_config.pop("variant", None)
        if run_variant not in solvers:
            raise ValueError(f"Unknown HC variant: {run_variant}")
        if run_variant == "stochastic" and run_config.get("max_iterations") is None:
            raise ValueError("Eksperimen stochastic memerlukan batas iterasi berhingga")
        run_config["objective"] = objective_name
        prepared.append((run_variant, run_config))
    print("\n--- Menjalankan 3 Eksperimen Hill-Climbing ---")
    for i, (seed, (run_variant, run_config)) in enumerate(zip(seeds, prepared), 1):
        prefix = "hc" if run_variant == "steepest_ascent" else f"hc_{run_variant}"
        print(f"Run {i}/3: {run_variant}, Seed {seed}...")
        result = solvers[run_variant](problem, config=run_config, seed=seed)
        stem = f"{prefix}_run_{i}_seed_{seed}"
        json_path = target_dir / f"{stem}.json"
        plot_path = target_dir / f"{stem}_objective.png"

        save_result(json_path, problem, result)
        plot_objective(result, plot_path)

        initial_obj = objective(problem, result.initial_state)
        records.append({
            "algorithm": prefix,
            "objective": objective_name,
            "experiment": f"run_{i}",
            "population": 1,
            "iterations": result.iterations,
            "seed": seed,
            "initial_objective": initial_obj,
            "final_objective": result.final_objective,
            "duration_seconds": round(result.duration_seconds, 6),
            "stop_reason": result.stop_reason,
            "stuck_count": 0,
        })
        print(f"  Selesai: obj={result.final_objective}, iter={result.iterations}, waktu={result.duration_seconds:.4f}s")

    _write_summary_csv(target_dir, records)
    return 0


def run_sa_experiments(
    problem: Problem,
    output_dir: Path | str,
    seeds: Sequence[int] = (42, 100, 2026),
    initial_temperature: float = 100.0,
    max_iterations: int = 1000,
    objective_name: str = "value",
) -> int:
    _validate_seeds(seeds)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    objective = get_objective(objective_name)
    config = {
        "initial_temperature": initial_temperature,
        "max_iterations": max_iterations,
        "objective": objective_name,
    }

    print(f"\n--- Menjalankan 3 Eksperimen Simulated Annealing ---")
    for i, seed in enumerate(seeds, 1):
        print(f"Run {i}/3: Seed {seed}...")
        result = simulated_annealing.solve(problem, config=config, seed=seed)
        stem = f"sa_run_{i}_seed_{seed}"
        json_path = target_dir / f"{stem}.json"
        obj_plot_path = target_dir / f"{stem}_objective.png"
        prob_plot_path = target_dir / f"{stem}_exp_delta_over_t.png"

        save_result(json_path, problem, result)
        plot_objective(result, obj_plot_path)
        plot_sa_probability(result, prob_plot_path)

        initial_obj = objective(problem, result.initial_state)
        stuck_count = result.metadata.get("stuck_count", 0)
        records.append({
            "algorithm": "sa",
            "objective": objective_name,
            "experiment": f"run_{i}",
            "population": 1,
            "iterations": result.iterations,
            "seed": seed,
            "initial_objective": initial_obj,
            "final_objective": result.final_objective,
            "duration_seconds": round(result.duration_seconds, 6),
            "stop_reason": result.stop_reason,
            "stuck_count": stuck_count,
        })
        print(f"  Selesai: obj={result.final_objective}, iter={result.iterations}, stuck={stuck_count}, waktu={result.duration_seconds:.4f}s")

    _write_summary_csv(target_dir, records)
    return 0


def run_ga_experiments(
    problem: Problem,
    output_dir: Path | str,
    seeds: Sequence[int] = (42, 100, 2026),
    control_pop: int = 20,
    iter_variants: Sequence[int] = (25, 50, 100),
    control_iter: int = 50,
    pop_variants: Sequence[int] = (10, 20, 40),
    objective_name: str = "value",
) -> int:
    _validate_seeds(seeds)
    _validate_variants(iter_variants, "iter_variants", 1)
    _validate_variants(pop_variants, "pop_variants", 2)
    integer(control_pop, "control_pop", 2)
    integer(control_iter, "control_iter", 1)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    objective = get_objective(objective_name)

    total_runs = len(iter_variants) * len(seeds) + len(pop_variants) * len(seeds)
    current_run = 0

    print(f"\n--- Menjalankan 18 Eksperimen Genetic Algorithm ---")
    # Kelompok A: Populasi Kontrol, Variasi Iterasi
    print(f"\n[Kelompok A] Populasi Kontrol = {control_pop}, Variasi Iterasi = {iter_variants}")
    for iters in iter_variants:
        for seed in seeds:
            current_run += 1
            print(f"Run {current_run}/{total_runs} (Kelompok A): Pop={control_pop}, Iter={iters}, Seed={seed}...")
            config = {"population_size": control_pop, "max_iterations": iters, "objective": objective_name}
            result = genetic_algorithm.solve(problem, config=config, seed=seed)

            stem = f"ga_kelompokA_pop_{control_pop}_iter_{iters}_seed_{seed}"
            json_path = target_dir / f"{stem}.json"
            plot_path = target_dir / f"{stem}_progress.png"

            save_result(json_path, problem, result)
            plot_ga_progress(result, plot_path)

            initial_obj = objective(problem, result.initial_state)
            records.append({
                "algorithm": "ga",
                "objective": objective_name,
                "experiment": f"KelompokA_pop_{control_pop}_iter_{iters}",
                "population": control_pop,
                "iterations": iters,
                "seed": seed,
                "initial_objective": initial_obj,
                "final_objective": result.final_objective,
                "duration_seconds": round(result.duration_seconds, 6),
                "stop_reason": result.stop_reason,
                "stuck_count": 0,
            })
            print(f"  Selesai: max_obj={result.final_objective}, waktu={result.duration_seconds:.4f}s")

    # Kelompok B: Iterasi Kontrol, Variasi Populasi
    print(f"\n[Kelompok B] Iterasi Kontrol = {control_iter}, Variasi Populasi = {pop_variants}")
    for pop in pop_variants:
        for seed in seeds:
            current_run += 1
            print(f"Run {current_run}/{total_runs} (Kelompok B): Pop={pop}, Iter={control_iter}, Seed={seed}...")
            config = {"population_size": pop, "max_iterations": control_iter, "objective": objective_name}
            result = genetic_algorithm.solve(problem, config=config, seed=seed)

            stem = f"ga_kelompokB_pop_{pop}_iter_{control_iter}_seed_{seed}"
            json_path = target_dir / f"{stem}.json"
            plot_path = target_dir / f"{stem}_progress.png"

            save_result(json_path, problem, result)
            plot_ga_progress(result, plot_path)

            initial_obj = objective(problem, result.initial_state)
            records.append({
                "algorithm": "ga",
                "objective": objective_name,
                "experiment": f"KelompokB_pop_{pop}_iter_{control_iter}",
                "population": pop,
                "iterations": control_iter,
                "seed": seed,
                "initial_objective": initial_obj,
                "final_objective": result.final_objective,
                "duration_seconds": round(result.duration_seconds, 6),
                "stop_reason": result.stop_reason,
                "stuck_count": 0,
            })
            print(f"  Selesai: max_obj={result.final_objective}, waktu={result.duration_seconds:.4f}s")

    _write_summary_csv(target_dir, records)
    return 0
