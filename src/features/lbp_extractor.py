import cv2
import numpy as np

class LBPExtractor:
    def __init__(self, num_points=8, radius=1):
        self.num_points = num_points
        self.radius = radius
        
    def extract(self, image):
        # Always convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray_image.shape
        
        # Pad image to handle borders
        padded = np.pad(gray_image, ((1, 1), (1, 1)), mode='constant')
        
        # Center image (original size)
        center = padded[1:-1, 1:-1]
        
        lbp_image = np.zeros((h, w), dtype=np.uint8)
        
        # Neighbors (dy, dx) in clockwise order starting from top-left
        # 0 1 2
        # 7   3
        # 6 5 4
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),
            (1, 1), (1, 0), (1, -1),
            (0, -1)
        ]
        
        # Vectorized LBP calculation
        for i, (dy, dx) in enumerate(neighbors):
            # Shifted image corresponding to the neighbor
            shifted = padded[1+dy:h+1+dy, 1+dx:w+1+dx]
            
            # Compare and add to LBP image
            # Bitwise shift: 1 << (7 - i)
            lbp_image += ((shifted >= center) * (1 << (7 - i))).astype(np.uint8)
        
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        
        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist
