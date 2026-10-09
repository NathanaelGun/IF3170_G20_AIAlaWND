import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import hill_climbing
from src.core import coordinate_map, objective, oriented_dimensions
from src.core.io import load_problem, save_result


def run_search(problem, config, seed):
    result = hill_climbing.solve(problem, config, seed)
    if objective(problem, result.final_state) != result.final_objective:
        raise ValueError("Algorithm returned an inconsistent final objective")
    return result


def print_state(problem, state, title):
    print(f"\n{title} | value={objective(problem, state)}")
    headers = ("Paket", "Status", "Posisi x,y,z", "Orientasi w,l,h", "Ukuran x,y,z",
               "Value", "Berat", "Fragile", "ETA")
    rows = []
    for package in problem.packages:
        placement = state.placements[package.id]
        rows.append((package.id, "Dalam" if placement.inside else "Luar",
                     str(placement.position), ",".join(placement.orientation),
                     str(oriented_dimensions(package, placement)), str(package.value),
                     str(package.weight), "Ya" if package.is_fragile else "Tidak", str(package.eta)))
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    print("\nPeta truk (. = kosong):")
    cells = coordinate_map(problem, state)
    width = max([len(str(problem.truck.dimensions.w - 1)), 1, *[len(p.id) for p in problem.packages]])
    for z, layer in enumerate(cells):
        print(f"z={z}")
        print(" y\\x " + " ".join(f"{x:>{width}}" for x in range(problem.truck.dimensions.w)))
        for y, row in enumerate(layer):
            print(f"{y:>4} " + " ".join(f"{cell or '.':>{width}}" for cell in row))


def print_result(problem, result):
    print_state(problem, result.initial_state, "State awal")
    print_state(problem, result.final_state, "State akhir")
    print(f"\n{result.algorithm.upper()}: objective={result.final_objective}, iterations={result.iterations}, "
          f"duration={result.duration_seconds:.6f}s, stop={result.stop_reason}")
    print("Riwayat objective: " + " -> ".join(f"{h['iteration']}:{h['objective']}" for h in result.history))


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
        text = input(f"{label} ({hint}): ").strip()
        if allow_unlimited and text.lower() == "tanpa":
            return None
        try:
            value = int(text)
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
        print("\n1. Hill-Climbing\n2. Simulated Annealing\n3. Genetic Algorithm\n4. Tiga eksperimen HC\n0. Keluar")
        choice = read_nonempty("Pilihan")
        if choice == "0":
            return None
        if choice == "4":
            return "experiments"
        if choice == "1":
            return "hc"
        if choice in ("2", "3"):
            print("Algoritma belum diimplementasikan. Pilih Hill-Climbing.")
            continue
        print("Pilih 0, 1, 2, 3, atau 4.")


def save_objective_plot(result, output):
    from experiments.run_hc import plot_result
    plot_path = output.with_name(output.stem + "_objective.png")
    plot_result(result, plot_path)
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
    if result.algorithm == "hc" and read_yes_no("Buat grafik objective HC?"):
        try:
            save_objective_plot(result, output)
        except (ImportError, OSError) as error:
            print(f"Grafik gagal dibuat: {error}. Instal matplotlib dan periksa lokasi output.")


def interactive():
    print("Optimasi Pengisian Paket Truk")
    input_path, problem = read_problem()
    d = problem.truck.dimensions
    print(f"Truk {d.w} x {d.l} x {d.h}, kapasitas {problem.truck.max_capacity}, {len(problem.packages)} paket")
    algorithm = choose_algorithm()
    if algorithm is None:
        return 0
    if algorithm == "experiments":
        from experiments.run_hc import main as run_experiments
        seeds = read_experiment_seeds()
        while True:
            output_dir = Path(read_nonempty("Folder hasil eksperimen"))
            try:
                return run_experiments(input_path=input_path, output_dir=output_dir, seeds=seeds)
            except OSError as error:
                print(f"Folder output tidak valid: {error}")
            except ImportError as error:
                print(f"Eksperimen gagal: {error}")
                return 1
    seed = read_integer("Seed")
    limit = read_integer("Maksimum iterasi", minimum=0, allow_unlimited=True)
    config = {}
    if limit is not None:
        config["max_iterations"] = limit
    print("\nMembuat initial state random dan menjalankan pencarian...")
    result = run_search(problem, config, seed)
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
