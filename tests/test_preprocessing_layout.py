import unittest

from scripts.preprocessing.prepare_public_layout import (
    PUBLIC_SEQUENCE_TO_RECORDING,
    recording_for_sequence,
)


class PreprocessingLayoutTests(unittest.TestCase):
    def test_public_sequence_mapping_matches_release_provenance(self):
        expected = {
            "avante_01": "avante_4",
            "avante_02": "avante_5",
            "avante_03": "avante_1",
            "avante_04": "avante_2",
            "avante_05": "avante_3",
            "carnival_01": "carnival_1",
            "carnival_02": "carnival_2",
            "carnival_03": "carnival_3",
            "malibu_01": "malibu_4",
            "malibu_02": "malibu_1",
            "malibu_03": "malibu_3",
            "malibu_04": "malibu_2",
            "sonata_01": "sonata_1",
            "xm3_01": "xm3_1",
        }
        self.assertEqual(PUBLIC_SEQUENCE_TO_RECORDING, expected)

    def test_raw_recording_name_is_accepted(self):
        self.assertEqual(recording_for_sequence("avante_1"), "avante_1")
        self.assertEqual(recording_for_sequence("carnival_3"), "carnival_3")

    def test_legacy_alias_is_accepted(self):
        alias = "2025_10_22_AVANTE_251022_indong_middle_0011_sync"
        self.assertEqual(recording_for_sequence(alias), "avante_1")


if __name__ == "__main__":
    unittest.main()
