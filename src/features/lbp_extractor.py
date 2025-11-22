import cv2
import numpy as np

class LBPExtractor:
    def __init__(self, num_points=8, radius=1):
        self.num_points = num_points
        self.radius = radius
        
    def extract(self, image):

        # Always convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create an empty LBP output image
        lbp_image = np.zeros_like(gray_image)

        # Compute LBP
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):

                center = gray_image[i, j]
                binary = []

                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1],
                    gray_image[i+1, j+1], gray_image[i+1, j], gray_image[i+1, j-1],
                    gray_image[i, j-1]
                ]

                for n in neighbors:
                    binary.append(1 if n >= center else 0)

                lbp_value = 0
                for index, bit in enumerate(binary):
                    lbp_value += bit << (7 - index)

                lbp_image[i, j] = lbp_value

        # Create histogram with 256 bins
       

        return hist
