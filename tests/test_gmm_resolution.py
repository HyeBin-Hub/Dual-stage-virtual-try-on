"""Real tensor checks for resolution support, TPS and checkpoint compatibility.

Requires PyTorch. Random inputs and weights do not verify trained image quality.
"""
import gc
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from GMM_networks import GMM, FeatureRegression, TpsGridGen, load_checkpoint, save_checkpoint


@unittest.skipIf(torch is None, 'PyTorch is required for model tensor tests')
class GMMResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_forward_at_both_resolutions(self):
        for height, width in [(256, 192), (512, 384)]:
            with self.subTest(resolution=(height, width)):
                torch.manual_seed(7)
                model = GMM(SimpleNamespace(fine_height=height, fine_width=width, grid_size=5)).eval()
                with torch.inference_mode():
                    grid, theta, warped, refinement = model(
                        torch.randn(1, 6, height, width), torch.randn(1, 3, height, width)
                    )
                self.assertEqual(tuple(grid.shape), (1, height, width, 2))
                self.assertEqual(tuple(theta.shape), (1, 50))
                self.assertEqual(tuple(warped.shape), (1, 3, height, width))
                self.assertEqual(tuple(refinement.shape), (1, 4, height, width))
                for tensor in [grid, theta, warped, refinement]:
                    self.assertTrue(torch.isfinite(tensor).all().item())
                self.assertEqual(model.regression.conv[0].in_channels, (height // 16) * (width // 16))
                self.assertEqual(model.regression.linear.in_features, 64 * (height // 64) * (width // 64))
                # Buffer registration must not break original state_dict keys.
                self.assertFalse(any(key.startswith('gridGen.') for key in model.state_dict()))
                del model, grid, theta, warped, refinement
                gc.collect()

    def test_regression_and_tps_backward_at_both_resolutions(self):
        for height, width in [(256, 192), (512, 384)]:
            with self.subTest(resolution=(height, width)):
                feature_size = (height // 16, width // 16)
                channels = feature_size[0] * feature_size[1]
                regression = FeatureRegression(channels, 50, use_cuda=False, feature_size=feature_size)
                correlation = torch.randn(1, channels, *feature_size, requires_grad=True)
                grid = TpsGridGen(height, width, grid_size=5, use_cuda=False)(regression(correlation))
                grid.square().mean().backward()
                self.assertTrue(torch.isfinite(correlation.grad).all().item())
                self.assertGreater(correlation.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(regression.conv[0].weight.grad).all().item())
                self.assertTrue(torch.isfinite(regression.linear.weight.grad).all().item())

    def test_zero_tps_offsets_are_identity_at_both_resolutions(self):
        for height, width in [(256, 192), (512, 384)]:
            with self.subTest(resolution=(height, width)):
                tps = TpsGridGen(height, width, grid_size=5, use_cuda=False)
                grid = tps(torch.zeros(2, 50))
                expected = torch.cat([tps.grid_X, tps.grid_Y], dim=3).expand(2, -1, -1, -1)
                torch.testing.assert_close(grid, expected, atol=1e-4, rtol=1e-4)
                self.assertEqual(tps.state_dict(), {})
                # Check that TPS tensors follow device / dtype movement as buffers.
                tps = tps.to(dtype=torch.float64)
                self.assertEqual(tps(torch.zeros(1, 50, dtype=torch.float64)).dtype, torch.float64)

    def test_checkpoint_round_trip_and_cross_resolution_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            for height, width in [(256, 192), (512, 384)]:
                with self.subTest(resolution=(height, width)):
                    features = (height // 16, width // 16)
                    model = FeatureRegression(features[0] * features[1], 50, use_cuda=False, feature_size=features)
                    checkpoint = Path(directory) / 'weights.pth'
                    saved = model.linear.weight.detach().clone()
                    parameter = model.linear.weight
                    save_checkpoint(model, checkpoint)
                    with torch.no_grad():
                        model.linear.weight.zero_()
                    load_checkpoint(model, checkpoint)
                    torch.testing.assert_close(model.linear.weight, saved, rtol=0, atol=0)
                    self.assertIs(model.linear.weight, parameter)
                    other_features = (32, 24) if height == 256 else (16, 12)
                    other = FeatureRegression(other_features[0] * other_features[1], 50,
                                              use_cuda=False, feature_size=other_features)
                    unchanged = other.conv[0].weight.detach().clone()
                    with self.assertRaisesRegex(ValueError, '256x192 GMM weights'):
                        load_checkpoint(other, checkpoint)
                    torch.testing.assert_close(other.conv[0].weight, unchanged, rtol=0, atol=0)

    def test_256_regression_preserves_original_weights_and_outputs(self):
        # Execute the archived class alone, avoiding its unused torchvision import.
        source = Path(__file__).resolve().parents[1] / 'archive/original/GMM_networks.py'
        original_class = next(node for node in ast.parse(source.read_text()).body
                              if isinstance(node, ast.ClassDef) and node.name == 'FeatureRegression')
        namespace = {'torch': torch, 'nn': torch.nn}
        exec(compile(ast.Module(body=[original_class], type_ignores=[]), str(source), 'exec'), namespace)
        original = namespace['FeatureRegression'](192, 50, use_cuda=False).eval()
        updated = FeatureRegression(192, 50, use_cuda=False, feature_size=(16, 12)).eval()
        self.assertEqual(list(original.state_dict()), list(updated.state_dict()))
        updated.load_state_dict(original.state_dict(), strict=True)
        correlation = torch.randn(1, 192, 16, 12)
        with torch.inference_mode():
            torch.testing.assert_close(updated(correlation), original(correlation), rtol=0, atol=0)

    def test_mismatched_input_is_rejected_before_feature_extraction(self):
        model = GMM(SimpleNamespace(fine_height=512, fine_width=384, grid_size=5))
        with self.assertRaisesRegex(ValueError, 'Expected person'):
            model(torch.zeros(1, 6, 256, 192), torch.zeros(1, 3, 512, 384))


if __name__ == '__main__':
    unittest.main()
