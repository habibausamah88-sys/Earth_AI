"""Data integrity tests; no model training. Run: python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from configs import config as cfg
from src import data

class DataIntegrity(unittest.TestCase):
    def test_dedup_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            for name in cfg.CLASS_NAMES:
                folder = root / name
                folder.mkdir()
                for i in range(24):
                    Image.fromarray(rng.integers(0,256,(8,8,3),dtype=np.uint8)).save(folder / f'{i:02}.png')
            # Same pixels, different filename, and a cross-label conflict.
            a,b = (root / c for c in cfg.CLASS_NAMES[:2])
            (a/'duplicate.png').write_bytes((a/'00.png').read_bytes())
            (b/'conflict.png').write_bytes((a/'01.png').read_bytes())
            (b/'broken.png').write_text('not an image')
            paths, labels, names = data.scan_dataset(root)
            self.assertEqual(len(paths),239)
            self.assertEqual(names,cfg.CLASS_NAMES)
            self.assertEqual(len(data.find_duplicates(root)),2)
            splits = data.stratified_split(paths,labels,cfg.SEED)
            self.assertTrue(data.leakage_check(splits))
            self.assertEqual(sum(len(v[0]) for v in splits.values()),239)
            self.assertEqual(len(data.dataset_signature(paths)),64)
            self.assertTrue(np.array_equal(splits['test'][0],data.stratified_split(paths,labels,cfg.SEED)['test'][0]))
            wrong = dict(splits)
            wrong['val'] = (np.append(splits['val'][0],splits['test'][0][0]),np.append(splits['val'][1],splits['test'][1][0]))
            with self.assertRaises(RuntimeError): data.leakage_check(wrong)

if __name__ == '__main__': unittest.main()
