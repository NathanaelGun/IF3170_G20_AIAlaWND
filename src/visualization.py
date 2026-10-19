"""Render saved search results as 3D figures without changing the global backend."""

import argparse
import sys
from math import ceil, sqrt
from pathlib import Path

from src.core import Problem, SearchResult, bounds, get_objective, oriented_dimensions
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
    axis.mouse_init(rotate_btn=1, zoom_btn=3)
    axis.grid(False)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        coordinate_axis.pane.set_alpha(0.04)


def _draw_packages(axis, problem, state, truck_index, colors, outside_positions):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    details = {}
    for package in problem.packages:
        placement = state.placements[package.id]
        if placement.truck_index != truck_index:
            continue
        low, high = bounds(package, placement)
        if truck_index is None:
            low = outside_positions[package.id]
            high = tuple(a + b for a, b in zip(low, oriented_dimensions(package, placement)))

        mesh = Poly3DCollection(box_faces(low, high), facecolors=colors[package.id],
                                edgecolors="#263445", linewidths=0.7,
                                linestyles="--" if package.is_fragile else "-", alpha=0.85)
        mesh.set_gid(f"package:{package.id}")
        mesh.set_picker(True)
        axis.add_collection3d(mesh)
        center = [(a + b) / 2 for a, b in zip(low, high)]
        label = package.id + (" [F]" if package.is_fragile else "")
        annotation = axis.text(*center, label, fontsize=8, ha="center", va="center", parse_math=False,
                               zorder=1000, visible=False, gid="package_label",
                               bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2})
        loc_label = f"Truk {truck_index + 1}" if truck_index is not None else "Luar"
        details[mesh] = (f"{label} | {loc_label} | posisi={placement.position}\n"
                         f"ukuran={oriented_dimensions(package, placement)} | "
                         f"orientasi={','.join(placement.orientation)} | value={package.value:g} | "
                         f"berat={package.weight:g} | ETA={package.eta}", annotation)
    if not details:
        axis.text2D(0.5, 0.5, "Tidak ada paket", transform=axis.transAxes,
                    ha="center", color="#526273")
    return details


def _connect_navigation(figure, axes):
    views = {axis: (axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d(),
                    axis.elev, axis.azim, axis.roll) for axis in axes}

    def zoom(axis, factor):
        if axis not in views:
            return
        toolbar = figure.canvas.toolbar
        if toolbar is not None:
            toolbar.push_current()
        for get_limits, set_limits in ((axis.get_xlim3d, axis.set_xlim3d),
                                       (axis.get_ylim3d, axis.set_ylim3d),
                                       (axis.get_zlim3d, axis.set_zlim3d)):
            low, high = get_limits()
            center, half = (low + high) / 2, (high - low) * factor / 2
            set_limits(center - half, center + half)
        if toolbar is not None:
            toolbar.push_current()
        figure.canvas.draw_idle()

    def on_scroll(event):
        if event.step:
            zoom(event.inaxes, 1.2 ** -max(-5, min(5, event.step)))

    def on_key(event):
        if event.key in ("+", "=", "-"):
            zoom(event.inaxes, 1 / 1.2 if event.key in ("+", "=") else 1.2)
        elif event.key == "r":
            for axis, (xlim, ylim, zlim, elev, azim, roll) in views.items():
                axis.set(xlim=xlim, ylim=ylim, zlim=zlim)
                axis.view_init(elev=elev, azim=azim, roll=roll)
            figure.canvas.draw_idle()

    figure.canvas.mpl_connect("scroll_event", on_scroll)
    figure.canvas.mpl_connect("key_press_event", on_key)


def build_result_figure(problem: Problem, result: SearchResult):
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    score = get_objective(result.parameters.get("objective", "value"))
    values = [score(problem, state) for state in (result.initial_state, result.final_state)]
    if values[1] != result.final_objective:
        raise ValueError("Final objective does not match the saved state")
    colors = {package.id: colormaps["tab20"](i % 20) for i, package in enumerate(problem.packages)}
    total_capacity = sum(t.max_capacity for t in problem.trucks)
    outside = [p for p in problem.packages
               if any(not state.placements[p.id].inside for state in (result.initial_state, result.final_state))]
    columns = max(1, ceil(sqrt(len(outside))))
    stride = max((max(p.dimensions.w, p.dimensions.l, p.dimensions.h) for p in outside), default=1) + 1
    outside_positions = {p.id: ((i % columns) * stride, (i // columns) * stride, 0)
                         for i, p in enumerate(outside)}
    outside_high = (columns * stride, max(1, ceil(len(outside) / columns)) * stride, stride)

    rows = len(problem.trucks) + 1
    figure = Figure(figsize=(12, 3.5 * rows + 2), facecolor="white")
    grid = figure.add_gridspec(rows, 2)
    figure.subplots_adjust(left=0.05, right=0.95, top=1 - 1.6 / figure.get_figheight(),
                           bottom=1.2 / figure.get_figheight(), wspace=0.15, hspace=0.28)
    variant = result.parameters.get("variant", result.algorithm)
    figure.suptitle(f"Hasil 3D | {result.algorithm.upper()} / {variant} | seed={result.seed}",
                    fontsize=15, fontweight="bold", y=0.98, parse_math=False)
    if len(problem.trucks) == 1:
        d = problem.trucks[0].dimensions
        truck_info = f"Truk {d.w} x {d.l} x {d.h} | kapasitas={problem.trucks[0].max_capacity:g}"
    else:
        truck_info = f"Armada ({len(problem.trucks)} truk) | total kapasitas={total_capacity:g}"
    figure.text(0.5, 1 - 0.65 / figure.get_figheight(), f"{truck_info} | iterasi={result.iterations} | durasi={result.duration_seconds:.4f}s",
                ha="center", fontsize=10)
    details = {}
    axes = [figure.add_subplot(grid[row, col], projection="3d") for row in range(rows) for col in range(2)]
    for col, (state, value, title) in enumerate(zip((result.initial_state, result.final_state),
                                                  values, ("State awal", "State akhir"))):
        loaded = [p for p in problem.packages if state.placements[p.id].inside]
        weight = sum(p.weight for p in loaded)
        objective_name = result.parameters.get("objective", "value")
        figure.text(0.27 if col == 0 else 0.73, 1 - 1.15 / figure.get_figheight(),
                    f"{title} | {objective_name}={value:g}\n"
                    f"Dalam: {len(loaded)} paket | berat={weight:g}/{total_capacity:g}",
                    ha="center", fontsize=11, fontweight="bold")
        for k, truck in enumerate(problem.trucks):
            inside_axis = axes[2 * k + col]
            tw, tl, th = truck.dimensions.w, truck.dimensions.l, truck.dimensions.h
            frame = Poly3DCollection(box_faces((0, 0, 0), (tw, tl, th)),
                                     facecolors=(0.75, 0.81, 0.87, 0.03),
                                     edgecolors="#718096", linewidths=0.8)
            frame.set_gid(f"truck_{k}")
            inside_axis.add_collection3d(frame)
            packages = [p for p in loaded if state.placements[p.id].truck_index == k]
            truck_weight = sum(p.weight for p in packages)
            inside_axis.set_title(f"Truk {k + 1} ({tw} x {tl} x {th}) | {len(packages)} paket\n"
                                  f"berat={truck_weight:g}/{truck.max_capacity:g}", fontsize=10)
            _set_limits(inside_axis, (0, 0, 0), (tw, tl, th))
            details.update(_draw_packages(inside_axis, problem, state, k, colors, outside_positions))

        outside_axis = axes[-2 + col]
        outside_axis.set_title(f"{title} - paket di luar: {len(problem.packages) - len(loaded)}", fontsize=10)
        _set_limits(outside_axis, (0, 0, 0), outside_high)
        details.update(_draw_packages(outside_axis, problem, state, None, colors, outside_positions))

    selected = figure.text(0.5, 0.65 / figure.get_figheight(), "Klik balok untuk melihat ID dan detail paket.",
                           ha="center", fontsize=9, parse_math=False, wrap=True, gid="package_details")
    figure.text(0.5, 0.15 / figure.get_figheight(), "Seret kiri: putar | scroll / +/-: zoom panel | R: reset semua | [F]: pecah-belah\n"
                "Paket luar disusun dalam grid tampilan; koordinat asli tersedia pada detail. Warna tetap antar-state.",
                ha="center", fontsize=9, color="#526273")

    def on_pick(event):
        if event.artist in details and event.mouseevent.button in (None, 1):
            for _, annotation in details.values():
                annotation.set_visible(False)
            text, annotation = details[event.artist]
            selected.set_text(text)
            annotation.set_visible(True)
            figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event", on_pick)
    _connect_navigation(figure, axes)
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
        tk.Label(root, text="Seret kiri: putar | scroll / +/-: zoom panel | R: reset semua | "
                 "Gunakan scrollbar kanan untuk panel lainnya.").pack(side=tk.TOP, fill=tk.X)
        info = tk.StringVar(value="Klik balok untuk melihat ID dan detail paket.")
        tk.Label(root, textvariable=info, height=2, anchor="w", justify=tk.LEFT).pack(side=tk.TOP, fill=tk.X)
        viewport = tk.Canvas(root, highlightthickness=0)
        scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=viewport.yview)
        viewport.configure(yscrollcommand=scrollbar.set)
        content = tk.Frame(viewport)
        content_window = viewport.create_window((0, 0), window=content, anchor="nw")
        viewport.bind("<Configure>", lambda event: viewport.itemconfigure(content_window, width=event.width))
        content.bind("<Configure>", lambda event: viewport.configure(scrollregion=viewport.bbox("all")))
        # An explicit Tk canvas works even after batch plots selected the Agg backend.
        # https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
        canvas = FigureCanvasTkAgg(figure, master=content)
        toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.get_tk_widget().configure(height=int(figure.get_figheight() * 85))
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X)
        selected = next(text for text in figure.texts if text.get_gid() == "package_details")
        canvas.mpl_connect("pick_event", lambda event: info.set(selected.get_text()))
        canvas.draw()
        toolbar.push_current()
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
