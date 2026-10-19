import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import genetic_algorithm, hill_climbing, hill_climbing_variants, simulated_annealing
from src.core import coordinate_map, objective, oriented_dimensions
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
    if objective(problem, result.final_state) != result.final_objective:
        raise ValueError("Algorithm returned an inconsistent final objective")
    return result


def print_state(problem, state, title):
    print(f"\n{title} | value={objective(problem, state)}")
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
    print_state(problem, result.initial_state, "State awal")
    print_state(problem, result.final_state, "State akhir")
    variant = result.parameters.get("variant", result.algorithm)
    print(f"\n{result.algorithm.upper()} ({variant}): objective={result.final_objective}, iterations={result.iterations}, "
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
        print(f"Frekuensi stuck di local optima (stagnasi): {stuck}")

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


def read_integer(label, minimum=None, allow_unlimited=False, default=None):
    hint = f"bilangan bulat >= {minimum}" if minimum is not None else "bilangan bulat"
    if allow_unlimited:
        hint += " atau 'tanpa'"
    if default is not None:
        hint += f", default {default}"
    while True:
        text = input(f"{label} ({hint}): ").strip()
        if not text and default is not None:
            return default
        if allow_unlimited and text.lower() == "tanpa":
            return None
        try:
            value = int(text)
            if minimum is None or value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Masukkan {hint}.")


def read_float(label, minimum=None, default=None):
    hint = f"bilangan real >= {minimum}" if minimum is not None else "bilangan real"
    if default is not None:
        hint += f", default {default}"
    while True:
        text = input(f"{label} ({hint}): ").strip()
        if not text and default is not None:
            return default
        try:
            value = float(text)
            if minimum is None or value >= minimum:
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
    print("Masukkan 3 seed untuk eksperimen (tekan Enter untuk memakai default: 42, 100, 2026):")
    first = input("Seed eksperimen 1 (default 42): ").strip()
    if not first:
        return (42, 100, 2026)
    try:
        seeds.append(int(first))
    except ValueError:
        seeds.append(42)

    while len(seeds) < 3:
        seed = read_integer(f"Seed eksperimen {len(seeds) + 1}")
        if seed in seeds:
            print("Seed eksperimen harus berbeda.")
            continue
        seeds.append(seed)
    return tuple(seeds)


def read_problem():
    while True:
        input_path = Path(read_nonempty("File input JSON"))
        try:
            return input_path, load_problem(input_path)
        except (ValueError, KeyError, TypeError, OSError) as error:
            print(f"Input tidak valid: {error}")


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
    choice = choose_algorithm()
    if choice == "0":
        return 0

    if choice in ("4", "8", "9"):
        seeds = read_experiment_seeds()
        default_folder = f"output/{'hc' if choice == '4' else 'sa' if choice == '8' else 'ga'}"
        while True:
            folder_input = input(f"Folder hasil eksperimen (default '{default_folder}'): ").strip()
            output_dir = Path(folder_input if folder_input else default_folder)
            try:
                if choice == "4":
                    from experiments.run_hc import main as run_experiments
                    return run_experiments(input_path=input_path, output_dir=output_dir, seeds=seeds)
                from experiments.runner import run_ga_experiments, run_sa_experiments
                if choice == "8":
                    return run_sa_experiments(problem, output_dir=output_dir, seeds=seeds)
                return run_ga_experiments(problem, output_dir=output_dir, seeds=seeds)
            except OSError as error:
                print(f"Folder output tidak valid: {error}")
            except ImportError as error:
                print(f"Eksperimen gagal: {error}")
                return 1
    config = {}
    algorithms = {"1": "hc", "2": "sa", "3": "ga", "5": "hc_sideways",
                  "6": "hc_random_restart", "7": "hc_stochastic"}
    algo_key = algorithms[choice]

    seed = read_integer("Seed acak", default=42)

    if algo_key.startswith("hc"):
        limit = read_integer("Maksimum iterasi", minimum=0, allow_unlimited=algo_key != "hc_stochastic",
                             default=1000 if algo_key == "hc_stochastic" else None)
        if limit is not None:
            config["max_iterations"] = limit
    elif algo_key == "sa":
        temp = read_float("Temperatur awal (T0)", minimum=0.001, default=100.0)
        limit = read_integer("Maksimum iterasi", minimum=1, default=1000)
        config["initial_temperature"] = temp
        config["max_iterations"] = limit
    elif algo_key == "ga":
        pop = read_integer("Jumlah populasi", minimum=2, default=20)
        limit = read_integer("Banyak iterasi (generasi)", minimum=1, default=50)
        config["population_size"] = pop
        config["max_iterations"] = limit

    if algo_key == "hc_sideways":
        config["max_sideways_moves"] = read_integer("Maksimum sideways move berturut-turut", minimum=0)
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
        print("\nProgram dihentikan.")
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Jalankan tanpa argumen: python src/main.py")
    raise SystemExit(main())
