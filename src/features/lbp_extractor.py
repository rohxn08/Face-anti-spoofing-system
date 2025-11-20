import numpy as np
from skimage.feature import local_binary_pattern
import cv2

class LBPExtractor:
    def __init__(self, P=8, R=1, method='uniform'):
        self.P = P
        self.R = R
        self.method = method

    def extract(self, image):
        """
        Extract LBP histogram from an image.
        :param image: Input image (BGR or Grayscale)
        :return: LBP histogram (normalized)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        lbp = local_binary_pattern(gray, self.P, self.R, self.method)
        
        # Calculate histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
