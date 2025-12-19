import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LBPExtractor:
    def __init__(self, num_points=8, radius=1, grid_x=4, grid_y=4):
        self.num_points = num_points
        self.radius = radius
        self.grid_x = grid_x
        self.grid_y = grid_y
        
    def extract(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        lbp = local_binary_pattern(gray, self.num_points, self.radius, method="uniform")
        h, w = gray.shape
        cell_h = h // self.grid_y
        cell_w = w // self.grid_x
        
        spatial_hist = []
        n_bins = int(lbp.max() + 1)
        if n_bins < self.num_points + 2:
            n_bins = self.num_points + 2
        
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                cell_lbp = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist, _ = np.histogram(cell_lbp.ravel(), bins=n_bins, range=(0, n_bins))
                spatial_hist.extend(hist)
        
        result = np.array(spatial_hist).astype("float")
        result /= (result.sum() + 1e-7)
        return result