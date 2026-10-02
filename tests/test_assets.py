import json
from pathlib import Path
import tempfile
import unittest

from tools.check_assets import check_assets


class AssetContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'pairs.txt').write_text('000001_0.jpg 000002_1.jpg\n')
        for folder, name in [('images', '000001_0.jpg'), ('image_parse', '000001_0.png'),
                             ('agnostic-v3.2', '000001_0.jpg'), ('skeletons', '000001_0.jpg'),
                             ('cloth_deformation', '000002_1.jpg'), ('cloth_mask_deformation', '000002_1.jpg')]:
            path = self.root / 'test' / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture for existence-only validation')
        pose = self.root / 'test/keypoints/000001_0.json'
        pose.parent.mkdir(parents=True)
        pose.write_text(json.dumps({'keypoints': [[1, 2, 0.9, 0]]}))

    def check(self, **kwargs):
        return check_assets(self.root, 'test', 'pairs.txt', inspect_images=False, **kwargs)

    def test_inference_does_not_require_training_mask(self):
        self.assertEqual(self.check(), [])

    def test_training_identifies_missing_ground_truth_mask(self):
        self.assertTrue(any('cloth_gt' in e for e in self.check(training=True)))

    def test_tom_warp_is_keyed_by_person(self):
        errors = self.check(stage='TOM')
        self.assertTrue(any('paired_warp_cloth-2/000001_0.jpg' in e for e in errors))
        self.assertFalse(any('paired_warp_cloth-2/000002_1.jpg' in e for e in errors))

    def test_missing_checkpoint_and_unsupported_resolution(self):
        errors = self.check(checkpoint=self.root / 'missing.pth', height=384, width=512)
        self.assertTrue(any('checkpoint' in e for e in errors))
        self.assertTrue(any('Unsupported resolution 384x512' in e for e in errors))

    def test_paper_resolution_is_supported(self):
        self.assertEqual(self.check(height=512, width=384), [])

    def test_invalid_keypoints_are_reported(self):
        (self.root / 'test/keypoints/000001_0.json').write_text('{"keypoints": [1, 2, 3]}')
        self.assertTrue(any('Invalid keypoints' in e for e in self.check()))

    def test_duplicate_people_and_malformed_pairs_are_reported(self):
        (self.root / 'pairs.txt').write_text('000001_0.jpg 000002_1.jpg\n000001_0.jpg 000002_1.jpg\nbad row extra\n')
        errors = self.check()
        self.assertTrue(any('Duplicate person' in e for e in errors))
        self.assertTrue(any('exactly two' in e for e in errors))


if __name__ == '__main__':
    unittest.main()
