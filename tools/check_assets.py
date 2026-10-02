"""Check the supplied dataset contract without importing PyTorch."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_config import validate_resolution


def check_assets(root, split, pairs_file, stage='GMM', training=False,
                 warp_dir='paired_warp_cloth-2', warp_key='person',
                 gt_mask_dir='cloth_gt', height=256, width=192,
                 inspect_images=True, checkpoint=None):
    root = Path(root)
    errors = []
    try:
        validate_resolution(height, width)
    except ValueError as exc:
        errors.append(str(exc))
    if checkpoint is not None and not Path(checkpoint).is_file():
        errors.append(f'Missing checkpoint: {checkpoint}')
    pairs_path = root / pairs_file
    if not pairs_path.is_file():
        return errors + [f'Missing pair list: {pairs_path}']
    rows = []
    for i, line in enumerate(pairs_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        names = line.split()
        if len(names) != 2:
            errors.append(f'Pair list line {i}: expected exactly two filenames.')
            continue
        if any(Path(n).name != n or n in {'.', '..'} for n in names):
            errors.append(f'Pair list line {i}: filenames must not contain directories.')
            continue
        rows.append(tuple(names))
    if not rows:
        errors.append('Pair list has no valid samples.')
    people = [r[0] for r in rows]
    if len(people) != len(set(people)):
        errors.append('Duplicate person filenames would overwrite inference output.')
    base = root / split
    visited = set()
    for person, garment in rows:
        paths = [('images', person), ('image_parse', person.replace('.jpg', '.png')),
                 ('agnostic-v3.2', person), ('skeletons', person.split('.')[0] + '.jpg')]
        if stage == 'GMM':
            paths += [('cloth_deformation', garment), ('cloth_mask_deformation', garment)]
        else:
            paths += [(warp_dir, person if warp_key == 'person' else garment)]
        if training:
            paths.append((gt_mask_dir, garment))
        for folder, name in paths:
            p = base / folder / name
            if p in visited:
                continue
            visited.add(p)
            if not p.is_file():
                errors.append(f'Missing image: {p}')
            elif inspect_images:
                try:
                    from PIL import Image
                    with Image.open(p) as img:
                        if img.size != (width, height):
                            errors.append(f'Wrong image size: {p}: {img.size}, expected {(width, height)}')
                        if folder == 'image_parse' and img.mode not in {'L', 'P'}:
                            errors.append(f'Parsing must contain label IDs, not RGB colors: {p}')
                        img.verify()
                except ImportError:
                    errors.append('Install Pillow to inspect image contents, or use --no-image-check.')
                    inspect_images = False
                except Exception as exc:
                    errors.append(f'Unreadable image: {p}: {exc}')
        pose = base / 'keypoints' / person.replace('.jpg', '.json')
        if not pose.is_file():
            errors.append(f'Missing keypoints: {pose}')
        else:
            try:
                points = json.loads(pose.read_text())['keypoints']
                flat = [v for row in points for v in row] if points and isinstance(points[0], list) else points
                if not flat or len(flat) % 4 or not all(isinstance(v, (int, float)) for v in flat):
                    raise ValueError('keypoints must be nonempty numeric groups of four')
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f'Invalid keypoints: {pose}: {exc}')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot', required=True)
    p.add_argument('--datamode', default='test')
    p.add_argument('--data_list', required=True)
    p.add_argument('--stage', choices=['GMM', 'TOM'], default='GMM')
    p.add_argument('--training', action='store_true')
    p.add_argument('--warp_cloth_dir', default='paired_warp_cloth-2')
    p.add_argument('--warp_cloth_key', choices=['person', 'garment'], default='person')
    p.add_argument('--gt_mask_dir', default='cloth_gt')
    p.add_argument('--fine_height', type=int, default=256)
    p.add_argument('--fine_width', type=int, default=192)
    p.add_argument('--checkpoint')
    p.add_argument('--no-image-check', action='store_true')
    p.add_argument('--check-env', action='store_true')
    a = p.parse_args()
    errors = check_assets(a.dataroot, a.datamode, a.data_list, a.stage, a.training,
                          a.warp_cloth_dir, a.warp_cloth_key, a.gt_mask_dir,
                          a.fine_height, a.fine_width, not a.no_image_check, a.checkpoint)
    if a.check_env:
        for module in ['torch', 'torchvision', 'numpy', 'PIL', 'tensorboardX']:
            if importlib.util.find_spec(module) is None:
                errors.append(f'Missing Python dependency: {module}')
    for error in errors[:40]:
        print('ERROR:', error)
    if len(errors) > 40:
        print(f'... {len(errors) - 40} additional errors')
    if errors:
        raise SystemExit(1)
    print('Asset contract passed. This check does not run the model or reproduce paper metrics.')


if __name__ == '__main__':
    main()
