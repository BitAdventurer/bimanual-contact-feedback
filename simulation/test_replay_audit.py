"""Boundary tests for the public native replay comparison. Uses the packaged reference."""

import unittest

import numpy as np

from .audit import compare_canonical, verification_passed
from .replay import criterion, read_reference


class CanonicalAuditTests(unittest.TestCase):
    def setUp(self):
        self.reference = read_reference()
        self.data = {
            k[6:]: v.copy()
            for k, v in self.reference.items()
            if k.startswith("smoke_") and k != "smoke_id"
        }

    def audit(self):
        return compare_canonical(self.data, self.reference)

    def test_identical_replay_passes_both_modes(self):
        report = self.audit()
        self.assertTrue(verification_passed([], report))
        self.assertTrue(verification_passed([], report, strict=True))
        self.assertTrue(report["legacy_max_abs_lt_1e_6"])

    def test_platform_sized_force_and_geometry_differences(self):
        # Representative submicrometre/submillinewton discrepancies on a hosted CPU.
        self.data["after_pad_N"] += 9.5044e-6
        self.data["after_clearance_mm"] += 6.9e-7
        self.data["after_depth_mm"] += 6.9e-7
        self.data["after_angle_rad"] += 4.5e-8
        result = self.audit()
        self.assertTrue(verification_passed([], result))
        self.assertFalse(verification_passed([], result, strict=True))
        self.assertFalse(result["legacy_max_abs_lt_1e_6"])
        self.assertGreater(result["fields"]["after_pad_N"]["max_abs"], 9e-6)

    def test_force_allowance_scales_per_element_and_does_not_mask_near_zero_error(self):
        key = "after_pad_N"
        self.reference["smoke_" + key].flat[0] = 0.0
        self.data[key].flat[0] = 1.1e-6
        result = self.audit()
        self.assertFalse(verification_passed([], result))
        self.assertEqual(result["channels"][key]["exceedances"], 1)

    def test_large_force_error_fails(self):
        self.data["before_pad_N"].flat[2] += 1e-3
        result = self.audit()
        self.assertFalse(verification_passed([], result))
        self.assertFalse(verification_passed([], result, strict=True))

    def test_sensor_force_and_torque_have_separate_units(self):
        self.data["mid_w"][0, 0, 0, 3] += 1.1e-7
        result = self.audit()
        self.assertEqual(result["channels"]["mid_w.force"]["exceedances"], 0)
        self.assertEqual(result["channels"]["mid_w.torque"]["exceedances"], 1)
        self.assertFalse(verification_passed([], result))

    def test_discrete_threshold_crossing_fails_even_with_tiny_float_difference(self):
        # One face sits just above the frozen 0.10 mm clearance threshold.
        values = dict(
            after_full=np.ones(1, bool),
            after_depth_mm=np.array([26.0]),
            after_clearance_mm=np.array([0.10000001]),
            after_angle_rad=np.array([0.01]),
            after_pad_N=np.ones((1, 2, 2)),
        )
        recorded = criterion(values, "after_")
        values["after_clearance_mm"] -= 2e-8
        self.assertFalse(np.array_equal(criterion(values, "after_"), recorded))
        self.assertFalse(verification_passed(["example"], self.audit()))

    def test_nonfinite_and_broadcastable_wrong_shape_are_rejected(self):
        self.data["root_w"] = self.data["root_w"][:1]
        with self.assertRaisesRegex(ValueError, "shape"):
            self.audit()
        self.data["root_w"] = self.reference["smoke_root_w"].copy()
        self.data["root_w"].flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.audit()

    def test_strict_distinguishes_signed_zero(self):
        self.reference["smoke_root_w"].flat[0] = +0.0
        self.data["root_w"].flat[0] = -0.0
        result = self.audit()
        self.assertTrue(verification_passed([], result))
        self.assertFalse(verification_passed([], result, strict=True))


if __name__ == "__main__":
    unittest.main()
