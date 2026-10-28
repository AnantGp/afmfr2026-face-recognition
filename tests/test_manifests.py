import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from afmfr.build_manifest import build_manifest
from afmfr.make_splits import split_manifest


class ManifestTests(unittest.TestCase):
    def test_dataset_prefix_prevents_identity_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for domain in ('a', 'b'):
                folder = root / domain / '001'
                folder.mkdir(parents=True)
                (folder / 'face.png').touch()
            manifest = root / 'manifest.csv'
            build_manifest(root, 'track1', manifest, 1, 0)
            with manifest.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['identity'] for r in rows}, {'a:001', 'b:001'})

    def test_splits_are_reproducible_and_holdout_identities_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'all.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image_path', 'identity', 'track', 'split'])
                writer.writeheader()
                for identity in range(10):
                    for image in range(6):
                        writer.writerow(dict(image_path=f'{identity}/{image}.png', identity=str(identity), track='track2', split='unsplit'))
            outputs = [root / 'first.csv', root / 'second.csv']
            for output in outputs:
                split_manifest(manifest, output, 20260502, 0.2, 2, 5)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            with outputs[0].open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            identities = {split: {r['identity'] for r in rows if r['split'] == split} for split in ('train', 'val_same', 'val_identity')}
            self.assertFalse(identities['train'] & identities['val_identity'])
            self.assertEqual(identities['train'], identities['val_same'])
            self.assertEqual(len(rows), 60)
            self.assertEqual(sum(r['split'] == 'val_identity' for r in rows), 12)


if __name__ == '__main__':
    unittest.main()
