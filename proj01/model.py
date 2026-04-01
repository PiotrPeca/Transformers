import numpy as np


def predict(image):
    """Simple thresholding segmentation model.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    gray = np.mean(image, axis=2)
    mask = (gray > 128).astype(np.uint8) * 255
    return mask
