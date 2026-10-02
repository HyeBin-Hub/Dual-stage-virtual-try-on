"""Exercise the real loader using synthetic files at each supported resolution."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ['torch', 'torchvision', 'PIL'])
if HAS_RUNTIME:
    from PIL import Image
    from cp_dataset import CPDataset


@unittest.skipUnless(HAS_RUNTIME, 'PyTorch, torchvision and Pillow are required for dataset tensor tests')
class DatasetResolutionTests(unittest.TestCase):
    def make_dataset(self, root, height, width, stage):
        (root / 'pairs.txt').write_text('person.jpg cloth.jpg\n')
        for folder, name, mode, color in [
            ('images', 'person.jpg', 'RGB', (128, 128, 128)),
            ('image_parse', 'person.png', 'L', 15),
            ('agnostic-v3.2', 'person.jpg', 'RGB', (128, 128, 128)),
            ('skeletons', 'person.jpg', 'RGB', (128, 128, 128)),
            ('cloth_deformation', 'cloth.jpg', 'RGB', (128, 128, 128)),
            ('cloth_mask_deformation', 'cloth.jpg', 'L', 255),
            ('cloth_gt', 'cloth.jpg', 'L', 255),
            ('paired_warp_cloth-2', 'person.jpg', 'RGB', (128, 128, 128)),
        ]:
            path = root / 'train' / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(mode, (width, height), color).save(path)
        pose = root / 'train/keypoints/person.json'
        pose.parent.mkdir(parents=True, exist_ok=True)
        pose.write_text(json.dumps({'keypoints': [[192, 256, 1, 0]]}))
        return CPDataset(SimpleNamespace(dataroot=str(root), datamode='train', stage=stage,
                                        data_list='pairs.txt', fine_height=height, fine_width=width,
                                        radius=2, require_gt_mask=True))

    def test_gmm_and_tom_data_at_both_resolutions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for height, width in [(256, 192), (512, 384)]:
                for stage in ['GMM', 'TOM']:
                    with self.subTest(resolution=(height, width), stage=stage):
                        sample = self.make_dataset(root, height, width, stage)[0]
                        self.assertEqual(tuple(sample['agnostic'].shape), (6, height, width))
                        for key in ['cloth', 'image', 'parse_cloth']:
                            self.assertEqual(tuple(sample[key].shape), (3, height, width))
                        self.assertEqual(tuple(sample['gt_cltoh_warp_mask'].shape), (1, height, width))
                        if stage == 'GMM':
                            self.assertEqual(tuple(sample['cloth_mask'].shape), (1, height, width))
                            self.assertEqual(tuple(sample['grid_image'].shape), (3, height, width))
                        # The 384x512 reference point must scale to the image centre.
                        self.assertGreater(sample['pose_image'][0, height // 2, width // 2].item(), 0.9)

    def test_inconsistent_data_resolution_reports_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.make_dataset(root, 512, 384, 'GMM')
            Image.new('L', (192, 256), 255).save(root / 'train/cloth_gt/cloth.jpg')
            with self.assertRaisesRegex(ValueError, 'cloth_gt/cloth.jpg.*expected'):
                dataset[0]


if __name__ == '__main__':
    unittest.main()
