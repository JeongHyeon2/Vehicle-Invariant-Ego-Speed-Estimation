import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from egospeed.data import PackedEgoSpeedDataset, resolve_dataset_roots
from egospeed.metrics import regression_metrics
from egospeed.models import EgoSpeedSmartROI, dann_weight


class ReleaseTests(unittest.TestCase):
    def test_dataset_root_resolution(self):
        data_root, mask_root = resolve_dataset_roots("dataset", None, None)
        self.assertEqual(data_root, Path("dataset/packed"))
        self.assertEqual(mask_root, Path("dataset/smartroi_masks"))
        with self.assertRaises(ValueError):
            resolve_dataset_roots("dataset", "packed", None)

    def test_dataset_root_environment_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "packed").mkdir()
            (root / "smartroi_masks").mkdir()
            with patch.dict("os.environ", {"EGOSPEED_DATASET_ROOT": str(root)}):
                data_root, mask_root = resolve_dataset_roots(None, None, None)
            self.assertEqual(data_root, root / "packed")
            self.assertEqual(mask_root, root / "smartroi_masks")

    def test_model_forward_shape(self):
        model = EgoSpeedSmartROI(num_vehicles=4).eval()
        clip = torch.randn(1, 3, 3, 8, 12)
        mask = torch.rand(1, 3, 8, 12)
        with torch.no_grad():
            output = model(clip, mask)
        self.assertEqual(output["speed_pred"].shape, (1,))
        self.assertEqual(output["z_speed"].shape, (1, 256))
        self.assertEqual(output["z_vehicle"].shape, (1, 256))

    def test_dann_schedule(self):
        self.assertEqual(dann_weight(0, 100, 0.1), 0.0)
        self.assertLess(0.09, dann_weight(100, 100, 0.1))
        self.assertLessEqual(dann_weight(100, 100, 0.1), 0.1)

    def test_metrics_filter(self):
        metrics = regression_metrics([0.1, 1.0, 10.0, 21.0], [9.0, 2.0, 8.0, 0.0])
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae_mps"], 1.5)
        self.assertTrue(np.isclose(metrics["rmse_mps"], np.sqrt(2.5)))

    def test_packed_dataset_without_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            sequence = "avante_01"
            data_root = tmp_path / "packed"
            mask_root = tmp_path / "smartroi_masks"
            sequence_root = data_root / sequence
            sequence_root.mkdir(parents=True)
            mask_root.mkdir()

            frames = torch.linspace(-1.0, 1.0, 20 * 8 * 12).reshape(20, 1, 8, 12).half()
            flow = torch.zeros(20, 2, 8, 12).half()
            speeds = torch.arange(20).float()
            torch.save(
                {"frames": frames, "flow_rate64": flow, "speeds_mps": speeds},
                sequence_root / "packed_depthnorm64.pt",
            )
            torch.save(
                {"masks_u8": torch.full((20, 8, 12), 255, dtype=torch.uint8)},
                mask_root / f"{sequence}__smartroi_mask_u8.pt",
            )

            dataset = PackedEgoSpeedDataset(
                [sequence],
                {"AVANTE": 0},
                data_root,
                mask_root,
                clip_length=3,
                stride=3,
                min_speed_mps=0.0,
            )
            clip, target, vehicle_id, mask = dataset[0]
            self.assertEqual(clip.shape, (3, 3, 8, 12))
            self.assertEqual(target.ndim, 0)
            self.assertEqual(vehicle_id.item(), 0)
            self.assertTrue(torch.all(mask == 1.0))

    def test_legacy_suv_vehicle_map_accepts_public_carnival_name(self):
        from egospeed.data import extract_vehicle

        self.assertEqual(extract_vehicle("carnival_01"), "CARNIVAL")
        self.assertEqual(extract_vehicle("2025_11_23_SUV_001_sync"), "CARNIVAL")


if __name__ == "__main__":
    unittest.main()
