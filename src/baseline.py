"""Initial Random Forest baseline"""

import numpy as np

from .data import IGNORE_LABEL


def pixel_features(image):
    """Represent each pixel by its grayscale intensity and a 3x3 local mean.

    Expect an image with pixel values from 0 to 255. Return an array shaped
    (height * width, 2), in row-major pixel order. Edge neighborhoods repeat
    the nearest image pixel, and both features are scaled to [0, 1].
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a nonempty grayscale image")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("Expected finite image values between 0 and 255")

    intensity = image.astype(np.float32) / 255.0
    height, width = intensity.shape
    padded = np.pad(intensity, 1, mode="edge")
    neighborhood_sum = np.zeros_like(intensity)
    for row in range(3):
        for column in range(3):
            neighborhood_sum += padded[row:row + height, column:column + width]
    neighborhood_mean = neighborhood_sum / 9.0
    return np.stack([intensity, neighborhood_mean], axis=-1).reshape(-1, 2)


def labeled_pixels(image, mask):
    """Return features and targets for labeled pixels of one image/mask pair.

    Later training code must call this only on training images. Validation and
    test images must never supply pixels used to fit the forest.
    """
    image = np.asarray(image)
    mask = np.asarray(mask)
    if mask.shape != image.shape:
        raise ValueError("Image and mask must have the same height and width")
    if not np.isin(mask, [0, 1, 2, 3, IGNORE_LABEL]).all():
        raise ValueError("Unexpected terrain label in mask")

    features = pixel_features(image)
    targets = mask.ravel()
    valid = targets != IGNORE_LABEL
    return features[valid], targets[valid].astype(np.int64)


def build_baseline():
    """Create an unfitted Random Forest using scikit-learn's defaults.

    No experiment settings are selected at this stage. Import scikit-learn
    when the estimator is requested so feature preparation stays independent.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier()
