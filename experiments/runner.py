import csv
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms import genetic_algorithm, hill_climbing, simulated_annealing
from src.core import Problem, SearchResult, objective
from src.core.io import save_result


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
    valid_points = [
        (h["iteration"], h["exp_delta_over_temperature"])
        for h in result.history
        if h["iteration"] > 0 and h.get("exp_delta_over_temperature") is not None
    ]
    if not valid_points:
        return

    iterations = [p[0] for p in valid_points]
    probs = [p[1] for p in valid_points]

    plt.figure(figsize=(8, 5))
    marker = "o" if len(iterations) <= 50 else None
    plt.plot(iterations, probs, marker=marker, color="#d62728", alpha=0.7, label=r"$e^{\Delta E / T}$")
    plt.title(rf"Simulated Annealing - Plot $e^{{\Delta E / T}}$ vs Iterasi (Seed {result.seed})")
    plt.xlabel("Banyak Iterasi")
    plt.ylabel(r"Probabilitas Penerimaan ($e^{\Delta E / T}$)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150, bbox_inches="tight")
    plt.close()


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
) -> int:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    config = {}
    if max_iterations is not None:
        config["max_iterations"] = max_iterations

    print(f"\n--- Menjalankan 3 Eksperimen Hill-Climbing (Steepest Ascent) ---")
    for i, seed in enumerate(seeds, 1):
        print(f"Run {i}/3: Seed {seed}...")
        result = hill_climbing.solve(problem, config=config, seed=seed)
        stem = f"hc_run_{i}_seed_{seed}"
        json_path = target_dir / f"{stem}.json"
        plot_path = target_dir / f"{stem}_objective.png"

        save_result(json_path, problem, result)
        plot_objective(result, plot_path)

        initial_obj = objective(problem, result.initial_state)
        records.append({
            "algorithm": "hc",
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
) -> int:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    config = {
        "initial_temperature": initial_temperature,
        "max_iterations": max_iterations,
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
) -> int:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []

    total_runs = len(iter_variants) * len(seeds) + len(pop_variants) * len(seeds)
    current_run = 0

    print(f"\n--- Menjalankan 18 Eksperimen Genetic Algorithm ---")
    # Kelompok A: Populasi Kontrol, Variasi Iterasi
    print(f"\n[Kelompok A] Populasi Kontrol = {control_pop}, Variasi Iterasi = {iter_variants}")
    for iters in iter_variants:
        for seed in seeds:
            current_run += 1
            print(f"Run {current_run}/{total_runs} (Kelompok A): Pop={control_pop}, Iter={iters}, Seed={seed}...")
            config = {"population_size": control_pop, "max_iterations": iters}
            result = genetic_algorithm.solve(problem, config=config, seed=seed)

            stem = f"ga_kelompokA_pop_{control_pop}_iter_{iters}_seed_{seed}"
            json_path = target_dir / f"{stem}.json"
            plot_path = target_dir / f"{stem}_progress.png"

            save_result(json_path, problem, result)
            plot_ga_progress(result, plot_path)

            initial_obj = objective(problem, result.initial_state)
            records.append({
                "algorithm": "ga",
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
            config = {"population_size": pop, "max_iterations": control_iter}
            result = genetic_algorithm.solve(problem, config=config, seed=seed)

            stem = f"ga_kelompokB_pop_{pop}_iter_{control_iter}_seed_{seed}"
            json_path = target_dir / f"{stem}.json"
            plot_path = target_dir / f"{stem}_progress.png"

            save_result(json_path, problem, result)
            plot_ga_progress(result, plot_path)

            initial_obj = objective(problem, result.initial_state)
            records.append({
                "algorithm": "ga",
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
