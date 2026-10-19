from pathlib import Path
from typing import Sequence

from src.core import Problem
from .runner import plot_objective as plot_result, run_hc_experiments


def main(input_path: Path | str, output_dir: Path | str, seeds: Sequence[int] = (42, 100, 2026),
         objective_name: str = "value") -> int:
    from src.core.io import load_problem
    problem = load_problem(input_path)
    return run_hc_experiments(problem, output_dir, seeds, objective_name=objective_name)


__all__ = ["plot_result", "run_hc_experiments", "main"]
