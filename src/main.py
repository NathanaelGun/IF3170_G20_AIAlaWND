import sys
from math import isfinite
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import genetic_algorithm, hill_climbing, hill_climbing_variants, simulated_annealing
from src.core import coordinate_map, get_objective, objective, oriented_dimensions
from src.core.io import load_problem, save_result


def run_search(problem, config, seed, algorithm="hc"):
    solvers = {"hc": hill_climbing.solve,
               "sa": simulated_annealing.solve,
               "ga": genetic_algorithm.solve,
               "hc_sideways": hill_climbing_variants.solve_sideways,
               "hc_random_restart": hill_climbing_variants.solve_random_restart,
               "hc_stochastic": hill_climbing_variants.solve_stochastic}
    if algorithm not in solvers:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    result = solvers[algorithm](problem, config, seed)
    if get_objective(config.get("objective", "value"))(problem, result.final_state) != result.final_objective:
        raise ValueError("Algorithm returned an inconsistent final objective")
    return result


def print_state(problem, state, title, objective_name="value"):
    score_text = "" if objective_name == "value" else \
        f" | {objective_name}={get_objective(objective_name)(problem, state)}"
    print(f"\n{title} | value={objective(problem, state)}{score_text}")
    headers = ("Paket", "Status", "Truk", "Posisi x,y,z", "Orientasi w,l,h", "Ukuran x,y,z",
               "Value", "Berat", "Fragile", "ETA")
    rows = []
    for package in problem.packages:
        placement = state.placements[package.id]
        rows.append((package.id, "Dalam" if placement.inside else "Luar",
                     str(placement.truck_index + 1) if placement.inside else "-",
                     str(placement.position), ",".join(placement.orientation),
                     str(oriented_dimensions(package, placement)), str(package.value),
                     str(package.weight), "Ya" if package.is_fragile else "Tidak", str(package.eta)))
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    truck_cells = coordinate_map(problem, state)
    for index, truck in enumerate(problem.trucks):
        print(f"\nPeta truk {index + 1} (. = kosong):")
        cells = truck_cells[index]
        width = max([len(str(truck.dimensions.w - 1)), 1, *[len(p.id) for p in problem.packages]])
        for z, layer in enumerate(cells):
            print(f"z={z}")
            print(" y\\x " + " ".join(f"{x:>{width}}" for x in range(truck.dimensions.w)))
            for y, row in enumerate(layer):
                print(f"{y:>4} " + " ".join(f"{cell or '.':>{width}}" for cell in row))

def print_result(problem, result):
    objective_name = result.parameters.get("objective", "value")
    print_state(problem, result.initial_state, "State awal", objective_name)
    print_state(problem, result.final_state, "State akhir", objective_name)
    variant = result.parameters.get("variant", result.algorithm)
    print(f"\n{result.algorithm.upper()} ({variant}): objective[{objective_name}]={result.final_objective}, "
          f"iterations={result.iterations}, "
          f"duration={result.duration_seconds:.6f}s, stop={result.stop_reason}")
    if variant == "sideways":
        print(f"Sideways move (total): {result.metadata['sideways_moves']}")
        print(f"Sideways berturut-turut: {result.metadata['consecutive_sideways_moves']}")
    elif variant == "random_restart":
        print(f"Jumlah restart: {result.metadata['restart_count']}")
        print(f"Iterasi per run (0 = run awal): {result.metadata['iterations_per_restart']}")
        print(f"Run dengan hasil terbaik: {result.metadata['best_restart']}")

    if result.algorithm == "sa":
        stuck = result.metadata.get("stuck_count", 0)
        print(f"Frekuensi masuk local optima (tanpa neighbor yang lebih baik): {stuck}")

    if result.algorithm == "ga" and result.history:
        last = result.history[-1]
        print(f"Populasi akhir - Maksimum: {last.get('max_objective')}, Rata-rata: {last.get('mean_objective'):.2f}")

    hist = result.history
    if len(hist) <= 30:
        print("Riwayat objective: " + " -> ".join(f"{h['iteration']}:{h['objective']}" for h in hist))
    else:
        first_part = " -> ".join(f"{h['iteration']}:{h['objective']}" for h in hist[:10])
        last_part = " -> ".join(f"{h['iteration']}:{h['objective']}" for h in hist[-10:])
        print(f"Riwayat objective ({len(hist)} entri): {first_part} -> ... -> {last_part}")


def read_nonempty(label):
    while True:
        text = input(f"{label}: ").strip()
        if text:
            return text
        print("Input tidak boleh kosong.")


def read_integer(label, minimum=None, allow_unlimited=False):
    hint = f"bilangan bulat >= {minimum}" if minimum is not None else "bilangan bulat"
    if allow_unlimited:
        hint += " atau 'tanpa'"
    while True:
        text = read_nonempty(f"{label} ({hint})")
        if allow_unlimited and text.lower() == "tanpa":
            return None
        try:
            value = int(text)
            if minimum is None or value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Masukkan {hint}.")


def read_float(label, minimum=None):
    hint = f"bilangan real >= {minimum}" if minimum is not None else "bilangan real"
    while True:
        text = read_nonempty(f"{label} ({hint})")
        try:
            value = float(text)
            if isfinite(value) and (minimum is None or value >= minimum):
                return value
        except ValueError:
            pass
        print(f"Masukkan {hint}.")


def read_yes_no(label):
    while True:
        answer = input(f"{label} [y/n]: ").strip().lower()
        if answer in ("y", "ya"):
            return True
        if answer in ("n", "tidak"):
            return False
        print("Masukkan y atau n.")


def read_experiment_seeds():
    seeds = []
    print("Masukkan 3 seed berbeda untuk eksperimen:")
    while len(seeds) < 3:
        seed = read_integer(f"Seed eksperimen {len(seeds) + 1}")
        if seed in seeds:
            print("Seed eksperimen harus berbeda.")
            continue
        seeds.append(seed)
    return tuple(seeds)


def read_experiment_variants(label, minimum):
    values = []
    while len(values) < 3:
        value = read_integer(f"{label} {len(values) + 1}", minimum=minimum)
        if value in values:
            print("Nilai variasi eksperimen harus berbeda.")
            continue
        values.append(value)
    return tuple(values)


def read_hc_experiment_runs(seeds):
    variants = {"1": "steepest_ascent", "2": "sideways",
                "3": "random_restart", "4": "stochastic"}
    runs = []
    for index, seed in enumerate(seeds, 1):
        print(f"\n=== Jenis Hill-Climbing untuk run {index}/{len(seeds)} (seed {seed}) ===")
        print("1. Steepest Ascent Hill-Climbing")
        print("2. Hill-Climbing with Sideways Move")
        print("3. Random Restart Hill-Climbing")
        print("4. Stochastic Hill-Climbing")
        while True:
            choice = read_nonempty("Pilihan jenis HC")
            if choice in variants:
                break
            print("Pilihan tidak valid. Silakan pilih 1 - 4.")
        variant = variants[choice]
        config = {"variant": variant,
                  "max_iterations": read_integer("Maksimum iterasi run ini", minimum=0,
                                                  allow_unlimited=variant != "stochastic")}
        if variant == "sideways":
            config["max_sideways_moves"] = read_integer("Maksimum total sideways move", minimum=0)
        elif variant == "random_restart":
            config["max_restarts"] = read_integer("Maksimum restart (setelah run awal)", minimum=0)
        runs.append(config)
    return tuple(runs)


def read_problem():
    while True:
        input_path = Path(read_nonempty("File input JSON"))
        try:
            return input_path, load_problem(input_path)
        except (ValueError, KeyError, TypeError, OSError) as error:
            print(f"Input tidak valid: {error}")


def choose_objective():
    while True:
        print("\n=== Pilihan Objective Function ===")
        print("1. value        : total value paket di dalam truk (objective spesifikasi)")
        print("2. eta_priority : total value / (1 + eta) paket di dalam truk (prioritas paket ber-ETA kecil)")
        choice = read_nonempty("Pilihan objective")
        if choice == "1":
            return "value"
        if choice == "2":
            return "eta_priority"
        print("Pilihan tidak valid. Silakan pilih 1 atau 2.")


def choose_algorithm():
    while True:
        print("\n=== Menu Pilihan Algoritma Local Search ===")
        print("1. Hill-Climbing (Steepest Ascent)")
        print("2. Simulated Annealing")
        print("3. Genetic Algorithm")
        print("4. Eksperimen Hill-Climbing (3 run)")
        print("5. Hill-Climbing with Sideways Move")
        print("6. Random Restart Hill-Climbing")
        print("7. Stochastic Hill-Climbing")
        print("8. Eksperimen Simulated Annealing (3 run)")
        print("9. Eksperimen Genetic Algorithm (18 run)")
        print("0. Keluar")
        choice = read_nonempty("Pilihan")
        if choice in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            return choice
        print("Pilihan tidak valid. Silakan pilih 0 - 9.")


def save_interactive_plots(result, output_path: Path):
    from experiments.runner import plot_ga_progress, plot_objective, plot_sa_probability
    stem = output_path.stem
    parent = output_path.parent

    if result.algorithm == "hc":
        save_objective_plot(result, output_path)
    elif result.algorithm == "sa":
        plot_path = parent / f"{stem}_objective.png"
        plot_objective(result, plot_path)
        prob_path = parent / f"{stem}_exp_delta_over_t.png"
        plot_sa_probability(result, prob_path)
        print(f"Grafik objective: {plot_path.resolve()}")
        print(f"Grafik exp(ΔE/T): {prob_path.resolve()}")
    elif result.algorithm == "ga":
        plot_path = parent / f"{stem}_progress.png"
        plot_ga_progress(result, plot_path)
        print(f"Grafik GA progress: {plot_path.resolve()}")


def save_objective_plot(result, output):
    from matplotlib.figure import Figure

    plot_path = output.with_name(output.stem + "_objective.png")
    figure = Figure(figsize=(8, 4.5))
    axis = figure.subplots()
    iterations = [entry["iteration"] for entry in result.history]
    axis.step(iterations, [entry["objective"] for entry in result.history], where="post", label="Objective")
    if result.parameters.get("variant") == "random_restart":
        axis.step(iterations, [entry["best_objective"] for entry in result.history],
                  where="post", linestyle="--", label="Terbaik sejauh ini")
        for entry in result.history:
            if entry.get("event") == "restart":
                axis.axvline(entry["iteration"], color="gray", alpha=0.3, linestyle=":")
    axis.set(xlabel="Iterasi", ylabel="Objective",
             title=f"Hill-Climbing: {result.parameters.get('variant', 'steepest_ascent')}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    print(f"Grafik: {plot_path.resolve()}")


def save_interactive_result(problem, result):
    while True:
        output = Path(read_nonempty("Simpan hasil JSON (mis. output/hasil.json)"))
        if output.suffix.lower() != ".json":
            print("Nama file hasil harus berakhiran .json.")
            continue
        try:
            save_result(output, problem, result)
            print(f"Saved: {output.resolve()}")
            break
        except OSError as error:
            print(f"Gagal menyimpan: {error}")

    if read_yes_no(f"Buat grafik dan buka jendela 3D untuk {result.algorithm.upper()}?"):
        try:
            save_interactive_plots(result, output)
        except (ImportError, OSError) as error:
            print(f"Grafik gagal dibuat: {error}. Pastikan matplotlib terinstal.")
        try:
            from src.visualization import save_result_3d, show_result_3d
            image_path = output.with_name(output.stem + "_states_3d.png")
            save_result_3d(problem, result, image_path)
            print(f"Gambar state 3D: {image_path.resolve()}")
            print("Tutup jendela 3D untuk kembali ke terminal.")
            show_result_3d(problem, result)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"Visualisasi 3D gagal: {error}. JSON hasil tetap tersimpan.")


def interactive():
    print("=== Optimasi Pengisian Paket Truk (IF3170 Local Search) ===")
    input_path, problem = read_problem()
    print(f"{len(problem.trucks)} truk, {len(problem.packages)} paket")
    for index, truck in enumerate(problem.trucks):
        d = truck.dimensions
        print(f"Truk {index + 1}: {d.w} x {d.l} x {d.h}, kapasitas {truck.max_capacity}")
    objective_name = choose_objective()
    choice = choose_algorithm()
    if choice == "0":
        return 0

    if choice in ("4", "8", "9"):
        seeds = read_experiment_seeds()
        if choice == "4":
            parameters = {"run_configs": read_hc_experiment_runs(seeds)}
        elif choice == "8":
            parameters = {"initial_temperature": read_float("Temperatur awal (T0)", minimum=0.001),
                          "max_iterations": read_integer("Maksimum iterasi per run", minimum=1)}
        else:
            print("Kelompok A: populasi tetap, 3 variasi iterasi. Kelompok B: iterasi tetap, 3 variasi populasi.")
            control_pop = read_integer("Populasi kontrol (kelompok A)", minimum=2)
            iter_variants = read_experiment_variants("Variasi iterasi (kelompok A)", minimum=1)
            control_iter = read_integer("Iterasi kontrol (kelompok B)", minimum=1)
            pop_variants = read_experiment_variants("Variasi populasi (kelompok B)", minimum=2)
            parameters = {"control_pop": control_pop, "iter_variants": iter_variants,
                          "control_iter": control_iter, "pop_variants": pop_variants}
        while True:
            output_dir = Path(read_nonempty("Folder hasil eksperimen"))
            try:
                from experiments.runner import run_hc_experiments, run_ga_experiments, run_sa_experiments
                if choice == "4":
                    return run_hc_experiments(problem, output_dir=output_dir, seeds=seeds,
                                           objective_name=objective_name, **parameters)
                if choice == "8":
                    return run_sa_experiments(problem, output_dir=output_dir, seeds=seeds,
                                              objective_name=objective_name, **parameters)
                return run_ga_experiments(problem, output_dir=output_dir, seeds=seeds,
                                          objective_name=objective_name, **parameters)
            except OSError as error:
                print(f"Folder output tidak valid: {error}")
            except ImportError as error:
                print(f"Eksperimen gagal: {error}")
                return 1
    config = {"objective": objective_name}
    algorithms = {"1": "hc", "2": "sa", "3": "ga", "5": "hc_sideways",
                  "6": "hc_random_restart", "7": "hc_stochastic"}
    algo_key = algorithms[choice]

    seed = read_integer("Seed acak")

    if algo_key.startswith("hc"):
        limit = read_integer("Maksimum iterasi", minimum=0, allow_unlimited=algo_key != "hc_stochastic")
        config["max_iterations"] = limit
    elif algo_key == "sa":
        temp = read_float("Temperatur awal (T0)", minimum=0.001)
        limit = read_integer("Maksimum iterasi", minimum=1)
        config["initial_temperature"] = temp
        config["max_iterations"] = limit
    elif algo_key == "ga":
        pop = read_integer("Jumlah populasi", minimum=2)
        limit = read_integer("Banyak iterasi (generasi)", minimum=1)
        config["population_size"] = pop
        config["max_iterations"] = limit

    if algo_key == "hc_sideways":
        config["max_sideways_moves"] = read_integer("Maksimum total sideways move", minimum=0)
    elif algo_key == "hc_random_restart":
        config["max_restarts"] = read_integer("Maksimum restart (setelah run awal)", minimum=0)
    print(f"\nMembuat initial state random dan menjalankan pencarian ({algo_key.upper()})...")
    result = run_search(problem, config, seed, algorithm=algo_key)
    print_result(problem, result)
    save_interactive_result(problem, result)
    return 0


def main() -> int:
    try:
        return interactive()
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Jalankan tanpa argumen: python src/main.py")
    raise SystemExit(main())
