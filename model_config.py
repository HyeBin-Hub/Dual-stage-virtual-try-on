"""Resolution contract shared by models, entry points and asset checks."""

SUPPORTED_RESOLUTIONS = ((256, 192), (512, 384))  # (height, width)


def validate_resolution(height, width):
    if (height, width) not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f'Unsupported resolution {height}x{width} (height x width). '
            'Use 256x192 or 512x384 with data and GMM weights at the same resolution.'
        )
    return height, width
