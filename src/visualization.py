"""Render saved search results as 3D figures without changing the global backend."""

import argparse
import sys
from pathlib import Path

from src.core import Problem, SearchResult, bounds, get_objective, objective, oriented_dimensions
from src.core.io import problem_from_dict, read_json, result_from_dict


def box_faces(low, high):
    x, y, z = low
    X, Y, Z = high
    return [
        [(x, y, z), (X, y, z), (X, Y, z), (x, Y, z)],
        [(x, y, Z), (X, y, Z), (X, Y, Z), (x, Y, Z)],
        [(x, y, z), (X, y, z), (X, y, Z), (x, y, Z)],
        [(x, Y, z), (X, Y, z), (X, Y, Z), (x, Y, Z)],
        [(x, y, z), (x, Y, z), (x, Y, Z), (x, y, Z)],
        [(X, y, z), (X, Y, z), (X, Y, Z), (X, y, Z)],
    ]


def _set_limits(axis, low, high):
    from matplotlib.ticker import MaxNLocator

    spans = [max(b - a, 1) for a, b in zip(low, high)]
    limits = [(a - span * 0.05, a + span * 1.05) for a, span in zip(low, spans)]
    axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x", ylabel="y", zlabel="z")
    # Matching the box aspect to data ranges keeps package dimensions proportional.
    axis.set_box_aspect(spans)
    axis.view_init(elev=25, azim=-60)
    axis.tick_params(labelsize=8)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=3, integer=True))


def _truck_offsets(problem: Problem) -> list[int]:
    offsets = []
    current_x = 0
    gap = 2
    for t in problem.trucks:
        offsets.append(current_x)
        current_x += t.dimensions.w + gap
    return offsets


def _draw_packages(axis, problem, state, inside, colors, truck_offsets: list[int] | None = None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if truck_offsets is None:
        truck_offsets = [0] * len(problem.trucks)

    details = {}
    for package in problem.packages:
        placement = state.placements[package.id]
        if placement.inside != inside:
            continue
        low, high = bounds(package, placement)
        if inside and placement.truck_index is not None:
            ox = truck_offsets[placement.truck_index]
            low = (low[0] + ox, low[1], low[2])
            high = (high[0] + ox, high[1], high[2])

        mesh = Poly3DCollection(box_faces(low, high), facecolors=colors[package.id],
                                edgecolors="#263445", linewidths=1,
                                linestyles="--" if package.is_fragile else "-", alpha=0.8)
        mesh.set_gid(f"package:{package.id}")
        mesh.set_picker(True)
        axis.add_collection3d(mesh)
        center = [(a + b) / 2 for a, b in zip(low, high)]
        label = package.id + (" [F]" if package.is_fragile else "")
        axis.text(*center, label, fontsize=8, ha="center", va="center", parse_math=False, zorder=1000,
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
        loc_label = f"Truk {placement.truck_index + 1}" if inside else "Luar"
        details[mesh] = (f"{label} | {loc_label} | posisi={placement.position} | "
                         f"ukuran={oriented_dimensions(package, placement)} | "
                         f"orientasi={','.join(placement.orientation)} | value={package.value:g} | "
                         f"berat={package.weight:g} | ETA={package.eta}")
    if not details:
        axis.text2D(0.5, 0.5, "Tidak ada paket", transform=axis.transAxes,
                    ha="center", color="#526273")
    return details


def build_result_figure(problem: Problem, result: SearchResult):
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    score = get_objective(result.parameters.get("objective", "value"))
    values = [score(problem, state) for state in (result.initial_state, result.final_state)]
    if values[1] != result.final_objective:
        raise ValueError("Final objective does not match the saved state")
    colors = {package.id: colormaps["tab20"](i % 20) for i, package in enumerate(problem.packages)}
    offsets = _truck_offsets(problem)
    total_w = offsets[-1] + problem.trucks[-1].dimensions.w
    max_l = max(t.dimensions.l for t in problem.trucks)
    max_h = max(t.dimensions.h for t in problem.trucks)
    total_capacity = sum(t.max_capacity for t in problem.trucks)

    outside_boxes = [bounds(package, state.placements[package.id])
                     for state in (result.initial_state, result.final_state)
                     for package in problem.packages if not state.placements[package.id].inside]
    outside_low = tuple(min(box[0][i] for box in outside_boxes) for i in range(3)) if outside_boxes else (0, 0, 0)
    outside_high = tuple(max(box[1][i] for box in outside_boxes) for i in range(3)) if outside_boxes else (total_w, max_l, max_h)

    figure = Figure(figsize=(13, 9), facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.2, 0.8))
    figure.subplots_adjust(left=0.04, right=0.96, top=0.83, bottom=0.14, wspace=0.08, hspace=0.25)
    variant = result.parameters.get("variant", result.algorithm)
    figure.suptitle(f"Hasil 3D | {result.algorithm.upper()} / {variant} | seed={result.seed}",
                    fontsize=15, fontweight="bold", y=0.98, parse_math=False)
    if len(problem.trucks) == 1:
        d = problem.trucks[0].dimensions
        truck_info = f"Truk {d.w} x {d.l} x {d.h} | kapasitas={problem.trucks[0].max_capacity:g}"
    else:
        truck_info = f"Armada ({len(problem.trucks)} truk) | total kapasitas={total_capacity:g}"
    figure.text(0.5, 0.935, f"{truck_info} | iterasi={result.iterations} | durasi={result.duration_seconds:.4f}s",
                ha="center", fontsize=10)
    details = {}
    # Row order is initial/final inside, followed by initial/final outside.
    axes = [figure.add_subplot(grid[row, col], projection="3d") for row in range(2) for col in range(2)]
    for col, (state, value, title) in enumerate(zip((result.initial_state, result.final_state),
                                                  values, ("State awal", "State akhir"))):
        inside_axis, outside_axis = axes[col], axes[col + 2]
        for k, truck in enumerate(problem.trucks):
            ox = offsets[k]
            tw, tl, th = truck.dimensions.w, truck.dimensions.l, truck.dimensions.h
            frame = Poly3DCollection(box_faces((ox, 0, 0), (ox + tw, tl, th)),
                                     facecolors=(0.75, 0.81, 0.87, 0.03),
                                     edgecolors="#718096", linewidths=0.8)
            frame.set_gid(f"truck_{k}")
            inside_axis.add_collection3d(frame)
            if len(problem.trucks) > 1:
                inside_axis.text(ox + tw / 2, tl / 2, th + 0.3, f"Truk {k+1}\n({tw}x{tl}x{th})",
                                 fontsize=7, ha="center", va="bottom", zorder=1000)

        loaded = [p for p in problem.packages if state.placements[p.id].inside]
        weight = sum(p.weight for p in loaded)
        if len(problem.trucks) == 1:
            inside_axis.set_title(f"{title} | value={value:g}\nDalam: {len(loaded)} paket | "
                                  f"berat={weight:g}/{problem.trucks[0].max_capacity:g}", fontsize=11)
        else:
            t_details = [
                f"T{i+1}:{sum(problem.package_by_id[k_id].weight for k_id, pl in state.placements.items() if pl.truck_index == i):g}/{t.max_capacity:g}"
                for i, t in enumerate(problem.trucks)
            ]
            inside_axis.set_title(f"{title} | value={value:g}\nDalam: {len(loaded)} paket | "
                                  f"berat={weight:g}/{total_capacity:g} ({', '.join(t_details)})", fontsize=9)

        outside_axis.set_title(f"{title} - paket di luar: {len(problem.packages) - len(loaded)}", fontsize=10)
        _set_limits(inside_axis, (0, 0, 0), (total_w, max_l, max_h))
        _set_limits(outside_axis, outside_low, outside_high)
        details.update(_draw_packages(inside_axis, problem, state, True, colors, truck_offsets=offsets))
        details.update(_draw_packages(outside_axis, problem, state, False, colors))

    selected = figure.text(0.5, 0.08, "Klik balok untuk melihat detail paket.", ha="center", fontsize=9,
                            parse_math=False, wrap=True)
    figure.text(0.5, 0.025, "Seret kiri: putar | seret kanan: zoom | toolbar Home: reset | [F]: pecah-belah\n"
                "Panel luar memakai koordinat asli. Warna paket konsisten antara state awal dan akhir.",
                ha="center", fontsize=9, color="#526273")

    def on_pick(event):
        if event.artist in details:
            selected.set_text(details[event.artist])
            figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event", on_pick)
    return figure


def save_result_3d(problem: Problem, result: SearchResult, path: Path | str) -> None:
    figure = build_result_figure(problem, result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150, facecolor="white")


def show_result_3d(problem: Problem, result: SearchResult) -> None:
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    figure = build_result_figure(problem, result)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError("Jendela 3D memerlukan desktop dan Tk. Gunakan --save gambar.png --no-show "
                           "untuk ekspor tanpa jendela.") from error
    try:
        root.title("Optimasi paket truk - visualisasi 3D")
        root.geometry(f"{min(1280, root.winfo_screenwidth() - 80)}x{min(900, root.winfo_screenheight() - 100)}")
        root.minsize(800, 600)
        root.protocol("WM_DELETE_WINDOW", root.quit)
        # An explicit Tk canvas works even after batch plots selected the Agg backend.
        # https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
        canvas = FigureCanvasTkAgg(figure, master=root)
        toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        print("Jendela 3D siap. Seret untuk memutar; tutup jendela untuk selesai.", flush=True)
        root.mainloop()
    except tk.TclError as error:
        raise RuntimeError(f"Jendela 3D tidak dapat ditampilkan: {error}") from error
    finally:
        root.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Buka state awal/akhir dari JSON hasil sebagai visualisasi 3D.")
    parser.add_argument("result", type=Path, help="File JSON hasil pencarian, termasuk data problem")
    parser.add_argument("--save", type=Path, help="Simpan gambar state 3D (mis. output/state.png)")
    parser.add_argument("--no-show", action="store_true", help="Hanya ekspor gambar tanpa membuka jendela")
    args = parser.parse_args(argv)
    if args.no_show and args.save is None:
        parser.error("--no-show memerlukan --save")
    try:
        data = read_json(args.result)
        problem, result = problem_from_dict(data["problem"]), result_from_dict(data)
        if args.save:
            save_result_3d(problem, result, args.save)
            print(f"Gambar 3D: {args.save.resolve()}")
        if not args.no_show:
            show_result_3d(problem, result)
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Visualisasi 3D gagal: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
