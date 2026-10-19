import unittest
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.core.models import Dimensions, Package, Placement, Problem, State, Truck
from src.core.geometry import is_valid, validation_errors, objective, objective_eta_priority
from src.core.search_space import empty_state, move_package, rotate_package, random_neighbor
from src.algorithms import hill_climbing, simulated_annealing, genetic_algorithm
from random import Random


class TestSpecCompliance(unittest.TestCase):
    def setUp(self):
        self.truck = Truck(Dimensions(10, 10, 10), max_capacity=100.0)
        self.pkg_normal_1 = Package("P1", Dimensions(3, 3, 2), value=50.0, weight=20.0, is_fragile=False, eta=3)
        self.pkg_normal_2 = Package("P2", Dimensions(3, 3, 2), value=70.0, weight=30.0, is_fragile=False, eta=0)
        self.pkg_fragile = Package("P_Fragile", Dimensions(4, 4, 2), value=90.0, weight=15.0, is_fragile=True, eta=1)
        self.pkg_heavy = Package("P_Heavy", Dimensions(2, 2, 2), value=10.0, weight=95.0, is_fragile=False, eta=2)

        self.problem = Problem(
            trucks=(self.truck,),
            packages=(self.pkg_normal_1, self.pkg_normal_2, self.pkg_fragile, self.pkg_heavy)
        )

    def test_01_boundary_constraint(self):
        """Uji 1: Paket yang diletakkan menembus dinding truk harus dideteksi invalid."""
        state = empty_state(self.problem)
        # Menempatkan paket P1 di koordinat x=9 (panjang 3) -> tembus batas x=10
        state = move_package(state, "P1", (9, 0, 0), truck_index=0)
        errors = validation_errors(self.problem, state)
        self.assertFalse(is_valid(self.problem, state))
        self.assertTrue(any("crosses truck boundary" in err for err in errors))

    def test_02_floating_constraint(self):
        """Uji 2: Kardus melayang (z > 0 tanpa tumpuan di bawahnya) harus dideteksi invalid."""
        state = empty_state(self.problem)
        # Menempatkan P1 melayang di z=3 tanpa alas di bawahnya
        state = move_package(state, "P1", (0, 0, 3), truck_index=0)
        errors = validation_errors(self.problem, state)
        self.assertFalse(is_valid(self.problem, state))
        self.assertTrue(any("no base point has support" in err for err in errors))

    def test_03_fragile_protection_constraint(self):
        """Uji 3: Barang pecah-belah (is_fragile=True) tidak boleh dijadikan tumpuan di bawah paket lain."""
        state = empty_state(self.problem)
        # Letakkan fragile di lantai z=0
        state = move_package(state, "P_Fragile", (0, 0, 0), truck_index=0)
        # Letakkan P1 tepat di atas fragile (z = 0 + 2 = 2)
        state = move_package(state, "P1", (0, 0, 2), truck_index=0)
        errors = validation_errors(self.problem, state)
        self.assertFalse(is_valid(self.problem, state))
        self.assertTrue(any("fragile package supports" in err for err in errors))

    def test_04_weight_capacity_constraint(self):
        """Uji 4: Total berat melebihi max_capacity truk harus dideteksi invalid."""
        state = empty_state(self.problem)
        # P1 (20) + P2 (30) + P_Heavy (95) = 145 > 100
        state = move_package(state, "P1", (0, 0, 0), truck_index=0)
        state = move_package(state, "P2", (4, 0, 0), truck_index=0)
        state = move_package(state, "P_Heavy", (0, 4, 0), truck_index=0)
        errors = validation_errors(self.problem, state)
        self.assertFalse(is_valid(self.problem, state))
        self.assertTrue(any("weight capacity exceeded" in err for err in errors))

    def test_05_non_overlapping_constraint(self):
        """Uji 5: Dua paket yang saling tumpang tindih (overlap) di dalam truk harus invalid."""
        state = empty_state(self.problem)
        # P1 dan P2 ditempatkan di posisi (0, 0, 0) yang sama
        state = move_package(state, "P1", (0, 0, 0), truck_index=0)
        state = move_package(state, "P2", (1, 1, 0), truck_index=0)
        errors = validation_errors(self.problem, state)
        self.assertFalse(is_valid(self.problem, state))
        self.assertTrue(any("packages overlap" in err for err in errors))

    def test_06_rotation_and_moves(self):
        """Uji 6: Rotasi 90 derajat menghasilkan orientasi yang valid sesuai 6 kemungkinan sumbu."""
        state = empty_state(self.problem)
        # Rotasi sumbu z
        rotated_z = rotate_package(state, "P1", "z")
        self.assertNotEqual(state.placements["P1"].orientation, rotated_z.placements["P1"].orientation)
        self.assertEqual(rotated_z.placements["P1"].orientation, ("y", "x", "z"))

    def test_07_ga_elitism_monotonicity(self):
        """Uji 7: Genetic Algorithm dengan elitisme menjamin max_objective non-decreasing per generasi."""
        result = genetic_algorithm.solve(self.problem, config={"population_size": 10, "max_iterations": 25}, seed=42)
        self.assertTrue(is_valid(self.problem, result.final_state))
        self.assertEqual(len(result.history), 26)
        for i in range(1, len(result.history)):
            curr_max = result.history[i]["max_objective"]
            prev_max = result.history[i - 1]["max_objective"]
            self.assertGreaterEqual(curr_max, prev_max, f"Max objective turun di iterasi {i}")
            self.assertLessEqual(result.history[i]["mean_objective"], curr_max + 1e-9)

    def test_08_sa_metropolis_probabilities(self):
        """Uji 8: Simulated Annealing mencatat acceptance_probability di [0.0, 1.0] dan stuck_count."""
        result = simulated_annealing.solve(self.problem, config={"initial_temperature": 50.0, "max_iterations": 100}, seed=42)
        self.assertTrue(is_valid(self.problem, result.final_state))
        self.assertIn("stuck_count", result.metadata)
        for entry in result.history:
            prob = entry.get("acceptance_probability")
            if prob is not None:
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_09_alternative_objective_eta(self):
        """Uji 9: Objective function alternatif (eta_priority) memprioritaskan paket ber-ETA rendah."""
        state = empty_state(self.problem)
        # Masukkan P1 (value=50, eta=3) -> obj_eta = 50 / (1 + 3) = 12.5
        # Masukkan P2 (value=70, eta=0) -> obj_eta = 70 / (1 + 0) = 70.0
        state = move_package(state, "P1", (0, 0, 0), truck_index=0)
        state = move_package(state, "P2", (4, 0, 0), truck_index=0)
        val_default = objective(self.problem, state)
        val_eta = objective_eta_priority(self.problem, state)
        self.assertEqual(val_default, 120.0)
        self.assertAlmostEqual(val_eta, 12.5 + 70.0, places=4)

    def test_10_multi_truck_support(self):
        """Uji 10: Multi-truck dapat memuat paket ke truk 1 dan truk 2 dengan validasi terpisah."""
        truck2 = Truck(Dimensions(8, 8, 8), max_capacity=150.0)
        multi_problem = Problem(
            trucks=(self.truck, truck2),
            packages=(self.pkg_normal_1, self.pkg_normal_2, self.pkg_fragile, self.pkg_heavy)
        )
        state = empty_state(multi_problem)
        # Tempatkan P1 di truk 1 (indeks 0) dan P2 di truk 2 (indeks 1)
        state = move_package(state, "P1", (0, 0, 0), truck_index=0)
        state = move_package(state, "P2", (0, 0, 0), truck_index=1)
        self.assertTrue(is_valid(multi_problem, state))
        self.assertEqual(state.placements["P1"].truck_index, 0)
        self.assertEqual(state.placements["P2"].truck_index, 1)
        self.assertEqual(objective(multi_problem, state), 120.0)


if __name__ == "__main__":
    unittest.main()

