import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment_io import file_hash, save_json


class ExperimentIOTests(unittest.TestCase):
    def test_hash_paths_and_strings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample'
            path.write_bytes(b'OCID\x00test')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(file_hash(path), expected)
            self.assertEqual(file_hash(str(path)), expected)

    def test_save_replaces_and_preserves_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            path.write_text('old')
            value = {'available': 88, 'trained': False}
            save_json(path, value)
            self.assertEqual(path.read_text(), json.dumps(value, indent=2, allow_nan=False))
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            save_json(str(path), value, trailing_newline=True)
            self.assertTrue(path.read_text().endswith('\n'))

    def test_nonfinite_does_not_destroy_previous_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            path.write_text('{"valid": true}')
            for invalid in (float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    save_json(path, {'bad': invalid})
                self.assertEqual(path.read_text(), '{"valid": true}')

    def test_compatibility_imports(self):
        from run_experiment2 import file_hash as legacy_hash, save_json as legacy_save
        from run_depth_scale import sha
        self.assertIs(legacy_hash, file_hash)
        self.assertIs(legacy_save, save_json)
        self.assertIs(sha, file_hash)


if __name__ == '__main__':
    unittest.main()
