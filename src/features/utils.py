import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_lbp_feature(image, radius=1, neighbors=8):
    """
    Standalone function to compute LBP histogram from an image.
    Useful for quick debugging without instantiating LBPExtractor.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros_like(gray)
    
    # Simple LBP implementation for visualization
    # Note: For production, use LBPExtractor which uses scikit-image's optimized implementation
    for r in range(0, gray.shape[0] - 2):
        for c in range(0, gray.shape[1] - 2):
            center = gray[r+1, c+1]
            code = 0
            code |= (gray[r, c] >= center) << 7
            code |= (gray[r, c+1] >= center) << 6
            code |= (gray[r, c+2] >= center) << 5
            code |= (gray[r+1, c+2] >= center) << 4
            code |= (gray[r+2, c+2] >= center) << 3
            code |= (gray[r+2, c+1] >= center) << 2
            code |= (gray[r+2, c] >= center) << 1
            code |= (gray[r+1, c] >= center) << 0
            lbp[r+1, c+1] = code
            
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, neighbors + 3), range=(0, neighbors + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist, lbp

def plot_lbp_histogram(image, title="LBP Histogram"):
    """
    Visualizes the LBP Texture and its Histogram side-by-side.
    """
    hist, lbp_image = compute_lbp_feature(image)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(lbp_image, cmap="gray")
    plt.title("LBP Texture")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.bar(range(len(hist)), hist)
    plt.title("Histogram")
    
    plt.show()

def normalize_image(image):
    """
    Normalizes image pixel values to [0, 1].
    """
    return image.astype("float32") / 255.0
