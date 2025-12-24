import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LBPExtractor:
    def __init__(self, num_points=24, radius=3, grid_x=4, grid_y=4):
        self.num_points = num_points
        self.radius = radius # INCREASED: Looks at material rather than noise
        self.grid_x = grid_x
        self.grid_y = grid_y
        
    def extract(self, image):
        if len(image.shape) == 3:
            img_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            # Fallback for grayscale input, though this extractor expects color for Cr/Cb focus
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        final_feature_vector = []

        # RESEARCH STEP: Use only Cr and Cb channels (skip Y which is channel 0) 
        # to focus on material chrominance distortion
        for channel_idx in [1, 2]: 
            channel = img_ycbcr[:, :, channel_idx]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            channel = clahe.apply(channel)

            # method="uniform" with higher radius captures macro-texture
            lbp = local_binary_pattern(channel, self.num_points, self.radius, method="uniform")
            
            h, w = channel.shape
            cell_h, cell_w = h // self.grid_y, w // self.grid_x
            n_bins = self.num_points + 2
            
            for i in range(self.grid_y):
                for j in range(self.grid_x):
                    cell_lbp = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    # density=True normalizes the histogram
                    hist, _ = np.histogram(cell_lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                    final_feature_vector.extend(hist)
        
        return np.array(final_feature_vector)