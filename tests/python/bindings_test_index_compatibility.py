import contextlib
import io
import os
import shutil
import struct
import tempfile
import unittest

import numpy as np

import hnswlib


class IndexCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hnswlib-compat-")
        self.dim = 8
        self.data = np.random.RandomState(7).rand(24, self.dim).astype("float32")
        self.labels = np.arange(len(self.data))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def build_index(self, name="index.bin"):
        index = hnswlib.Index(space="l2", dim=self.dim)
        index.init_index(max_elements=32, M=8, ef_construction=50)
        index.add_items(self.data, self.labels)
        path = self.path(name)
        index.save_index(path)
        return index, path

    def load_without_warning_noise(self, path, **kwargs):
        index = hnswlib.Index(space="l2", dim=self.dim)
        with contextlib.redirect_stderr(io.StringIO()):
            index.load_index(path, **kwargs)
        return index

    def legacy_copy(self, source, destination):
        with open(source, "rb") as src:
            raw = src.read()
        start = hnswlib.LEGACY_HEADER_SIZE
        header_end = start + hnswlib.COMPAT_HEADER_SIZE
        footer_start = len(raw) - hnswlib.COMPAT_FOOTER_SIZE
        with open(destination, "wb") as dst:
            dst.write(raw[:start] + raw[header_end:footer_start])

    def rewrite_metadata_u32(self, source, destination, field_offset, value):
        with open(source, "rb") as src:
            raw = bytearray(src.read())
        header_payload = (
            hnswlib.LEGACY_HEADER_SIZE + len(b"HNSW-IDX-COMPAT\x00")
        )
        footer_payload = (
            len(raw) - hnswlib.COMPAT_FOOTER_SIZE
            + len(b"HNSW-IDX-FOOTER\x00")
        )
        for base in (header_payload, footer_payload):
            struct.pack_into("<I", raw, base + field_offset, value)
        with open(destination, "wb") as dst:
            dst.write(raw)

    def test_same_version_roundtrip_preserves_query(self):
        original, path = self.build_index()
        expected_labels, expected_distances = original.knn_query(self.data, k=3)

        report = hnswlib.inspect_index_file(path, "l2", self.dim)
        self.assertEqual(report["status"], "compatible")
        self.assertTrue(report["loadable"])
        self.assertFalse(report["requires_rebuild"])
        self.assertEqual(
            report["metadata"]["format_version"],
            hnswlib.INDEX_FORMAT_VERSION,
        )
        self.assertEqual(report["metadata"]["space"], "l2")
        self.assertEqual(report["metadata"]["dim"], self.dim)

        loaded = self.load_without_warning_noise(path)
        actual_labels, actual_distances = loaded.knn_query(self.data, k=3)
        np.testing.assert_array_equal(actual_labels, expected_labels)
        np.testing.assert_allclose(actual_distances, expected_distances)

    def test_legacy_file_loads_and_can_be_upgraded(self):
        original, current_path = self.build_index()
        expected_labels, expected_distances = original.knn_query(self.data, k=3)
        legacy_path = self.path("legacy.bin")
        self.legacy_copy(current_path, legacy_path)

        report = hnswlib.inspect_index_file(legacy_path, "l2", self.dim)
        self.assertEqual(report["status"], "upgradeable")
        self.assertTrue(report["legacy"])
        self.assertTrue(report["loadable"])

        loaded = self.load_without_warning_noise(legacy_path)
        actual_labels, actual_distances = loaded.knn_query(self.data, k=3)
        np.testing.assert_array_equal(actual_labels, expected_labels)
        np.testing.assert_allclose(actual_distances, expected_distances)

        upgraded_path = self.path("upgraded.bin")
        loaded.save_index(upgraded_path)
        upgraded_report = hnswlib.inspect_index_file(upgraded_path)
        self.assertEqual(upgraded_report["status"], "compatible")
        upgraded = self.load_without_warning_noise(upgraded_path)
        upgraded_labels, _ = upgraded.knn_query(self.data, k=3)
        np.testing.assert_array_equal(upgraded_labels, expected_labels)

    def test_parameter_conflicts_are_explicit(self):
        _, path = self.build_index()

        wrong_dim = hnswlib.Index(space="l2", dim=self.dim + 1)
        with self.assertRaisesRegex(RuntimeError, "parameter conflict.*dim"):
            wrong_dim.load_index(path)

        wrong_space = hnswlib.Index(space="cosine", dim=self.dim)
        with self.assertRaisesRegex(RuntimeError, "parameter conflict.*space"):
            wrong_space.load_index(path)

    def test_truncated_file_is_rejected_before_load(self):
        _, path = self.build_index()
        truncated = self.path("truncated.bin")
        with open(path, "rb") as src:
            raw = src.read()
        with open(truncated, "wb") as dst:
            dst.write(raw[:-32])

        report = hnswlib.inspect_index_file(truncated)
        self.assertTrue(report["truncated"])
        self.assertEqual(report["status"], "requires_rebuild")
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            hnswlib.Index(space="l2", dim=self.dim).load_index(truncated)

    def test_newer_index_is_rejected_and_check_can_be_disabled(self):
        _, path = self.build_index()
        future_path = self.path("future.bin")
        future_version = hnswlib.INDEX_READER_VERSION + 1
        self.rewrite_metadata_u32(
            path, future_path, field_offset=4, value=future_version
        )
        self.rewrite_metadata_u32(
            future_path, future_path, field_offset=8, value=future_version
        )

        report = hnswlib.inspect_index_file(future_path)
        self.assertEqual(report["status"], "requires_rebuild")
        self.assertIn("requires reader", report["reason"])
        with self.assertRaisesRegex(RuntimeError, "requires reader"):
            hnswlib.Index(space="l2", dim=self.dim).load_index(future_path)

        loaded = self.load_without_warning_noise(
            future_path, check_compatibility=False
        )
        labels, _ = loaded.knn_query(self.data, k=1)
        np.testing.assert_array_equal(labels.reshape(-1), self.labels)

    def test_byte_order_mismatch_is_explicit(self):
        _, path = self.build_index()
        swapped_path = self.path("swapped_endian.bin")
        current_endian = hnswlib.inspect_index_file(path)["metadata"]["byte_order_id"]
        other_endian = 0x04030201 if current_endian == 0x01020304 else 0x01020304
        self.rewrite_metadata_u32(
            path, swapped_path, field_offset=16, value=other_endian
        )
        report = hnswlib.inspect_index_file(swapped_path)
        self.assertTrue(report["endian_mismatch"])
        self.assertEqual(report["status"], "requires_rebuild")
        with self.assertRaisesRegex(RuntimeError, "byte order"):
            hnswlib.Index(space="l2", dim=self.dim).load_index(swapped_path)

    def test_alignment_difference_warns_but_loads(self):
        _, path = self.build_index()
        aligned_path = self.path("alignment_metadata.bin")
        self.rewrite_metadata_u32(
            path, aligned_path, field_offset=28, value=32
        )
        report = hnswlib.inspect_index_file(aligned_path)
        self.assertEqual(report["status"], "compatible")
        self.assertTrue(
            any("alignment" in warning for warning in report["warnings"])
        )
        loaded = self.load_without_warning_noise(aligned_path)
        self.assertTrue(
            any(
                "alignment" in warning
                for warning in loaded.get_compatibility_warnings()
            )
        )
        labels, _ = loaded.knn_query(self.data, k=1)
        np.testing.assert_array_equal(labels.reshape(-1), self.labels)

    def test_batch_scan_returns_actionable_categories(self):
        _, current_path = self.build_index("compatible.bin")
        legacy_path = self.path("legacy.bin")
        self.legacy_copy(current_path, legacy_path)
        truncated_path = self.path("truncated.bin")
        with open(current_path, "rb") as src:
            raw = src.read()
        with open(truncated_path, "wb") as dst:
            dst.write(raw[:-32])

        report = hnswlib.scan_index_files(
            [current_path, legacy_path, truncated_path], "l2", self.dim
        )
        self.assertEqual(report["total"], 3)
        upgradeable_paths = {item["path"] for item in report["upgradeable"]}
        rebuild_paths = {item["path"] for item in report["rebuild_required"]}
        self.assertEqual(upgradeable_paths, {current_path, legacy_path})
        self.assertEqual(rebuild_paths, {truncated_path})
        for item in report["upgradeable"] + report["rebuild_required"]:
            self.assertIn("reason", item)
            self.assertIn("action", item)


if __name__ == "__main__":
    unittest.main()
