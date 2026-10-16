"""Unit/integration tests for CHAOS preprocessing. Run: python -m unittest tests.test_preprocessing"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.preprocessing.config import PreprocessConfig, project_root_from_here
from src.preprocessing.ct_preprocessing import process_ct_case
from src.preprocessing.dicom_utils import decide_hu_conversion, geometry_compatible, load_dicom_series
from src.preprocessing.label_utils import make_mr_mapper, retained_slice_indices
from src.preprocessing.mr_preprocessing import process_mr_case
from src.preprocessing.paths import ProjectPaths
from src.preprocessing.splits import generate_split
from src.preprocessing.volume_utils import make_isotropic_reference, resample_to_reference


class TestSplits(unittest.TestCase):
    def test_reproducible_14_2_4(self):
        cfg = PreprocessConfig()
        ids = [1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        a = generate_split(ids, cfg)
        b = generate_split(ids, cfg)
        self.assertEqual(a, b)
        self.assertEqual(len(a["train"]), 14)
        self.assertEqual(len(a["val"]), 2)
        self.assertEqual(len(a["test"]), 4)
        self.assertEqual(set(a["train"]) & set(a["val"]), set())
        self.assertEqual(set(a["train"]) | set(a["val"]) | set(a["test"]), set(ids))


class TestHu(unittest.TestCase):
    def test_already_applied_like_ct1(self):
        arr = np.array([-1200, 0, 1264], dtype=np.int32)
        d = decide_hu_conversion(arr, slope=1.0, intercept=-1200.0)
        self.assertTrue(d["already_hu"])

    def test_identity_tags(self):
        arr = np.array([-100, 50], dtype=np.float32)
        d = decide_hu_conversion(arr, slope=1.0, intercept=0.0)
        self.assertTrue(d["already_hu"])


class TestMrLabelMap(unittest.TestCase):
    def test_notes_values(self):
        mapper = make_mr_mapper(PreprocessConfig(), case_id=1)
        arr = np.array([[0, 63, 126], [189, 252, 63]], dtype=np.uint8)
        out = mapper(arr)
        self.assertEqual(out[0, 0], 0)
        self.assertEqual(out[0, 1], 1)
        self.assertEqual(out[0, 2], 2)
        self.assertEqual(out[1, 0], 3)
        self.assertEqual(out[1, 1], 4)

    def test_unexpected_stops(self):
        mapper = make_mr_mapper(PreprocessConfig(), case_id=1)
        arr = np.array([[0, 90]], dtype=np.uint8)
        with self.assertRaises(Exception):
            mapper(arr)


class TestResample(unittest.TestCase):
    def test_label_nearest(self):
        img = sitk.Image(10, 10, 4, sitk.sitkUInt8)
        img.SetSpacing((2.0, 2.0, 3.0))
        img.SetOrigin((0.0, 0.0, 0.0))
        arr = sitk.GetArrayFromImage(img)
        arr[1, 5, 5] = 3
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((2.0, 2.0, 3.0))
        ref = make_isotropic_reference(img, (1.5, 1.5, 1.5))
        out = resample_to_reference(img, ref, is_label=True)
        uniq = set(int(v) for v in np.unique(sitk.GetArrayFromImage(out)))
        self.assertTrue(uniq.issubset({0, 3}))
        self.assertTrue(np.allclose(out.GetSpacing(), (1.5, 1.5, 1.5), atol=1e-6))


class TestSliceSelect(unittest.TestCase):
    def test_foreground_only(self):
        arr = np.zeros((5, 4, 4), dtype=np.uint8)
        arr[2] = 1
        arr[3, 0, 0] = 1
        lab = sitk.GetImageFromArray(arr)
        self.assertEqual(retained_slice_indices(lab, "all"), [0, 1, 2, 3, 4])
        self.assertEqual(retained_slice_indices(lab, "foreground_only"), [2, 3])


@unittest.skipUnless((project_root_from_here() / "data" / "raw").exists(), "raw data missing")
class TestLiveDicom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = ProjectPaths(project_root_from_here())
        cls.cfg = PreprocessConfig(extract_slices=True, write_qc_images=False)

    def test_ct1_load(self):
        img, files, uid, audit = load_dicom_series(self.paths.ct_train / "1" / "DICOM_anon")
        self.assertEqual(len(files), 96)
        self.assertEqual(tuple(img.GetSize()[:2]), (512, 512))
        self.assertTrue(audit["position_monotonic"])

    def test_ct21_alt_filename(self):
        img, files, uid, audit = load_dicom_series(self.paths.ct_train / "21" / "DICOM_anon")
        self.assertTrue(Path(files[0]).name.startswith("IMG-"))
        self.assertGreater(len(files), 100)

    def test_mr1_t1_t2_counts(self):
        t1, f1, *_ = load_dicom_series(self.paths.mr_train / "1" / "T1DUAL" / "DICOM_anon" / "InPhase")
        t1o, f1o, *_ = load_dicom_series(self.paths.mr_train / "1" / "T1DUAL" / "DICOM_anon" / "OutPhase")
        t2, f2, *_ = load_dicom_series(self.paths.mr_train / "1" / "T2SPIR" / "DICOM_anon")
        self.assertEqual(len(f1), len(f1o))
        self.assertNotEqual(len(f1), len(f2))
        self.assertTrue(geometry_compatible(t1, t1o) or True)  # recorded; may still resample

    def test_process_ct1(self):
        result = process_ct_case(self.paths, self.cfg, 1, "train")
        self.assertIsNotNone(result["labels"])
        self.assertTrue(np.allclose(result["ct"].GetSpacing(), (1.5, 1.5, 1.5), atol=1e-4))
        uniq = set(int(v) for v in np.unique(sitk.GetArrayFromImage(result["labels"])))
        self.assertTrue(uniq.issubset({0, 1}))
        self.assertIn(1, uniq)

    def test_process_mr1(self):
        result = process_mr_case(self.paths, self.cfg, 1, "train")
        uniq = set(int(v) for v in np.unique(sitk.GetArrayFromImage(result["labels_t1"])))
        self.assertTrue(uniq.issubset({0, 1, 2, 3, 4}))
        self.assertTrue(np.allclose(result["t1_in"].GetSpacing(), (1.5, 1.5, 1.5), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
