import os
import cv2
import joblib
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor

class InferencePipeline:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default model path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(base_dir, 'src', 'models', 'svm_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = joblib.load(model_path)
        self.extractor = LBPExtractor()
        
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        hist = self.extractor.extract(img)
        # Reshape for single sample
        hist = hist.reshape(1, -1)
        
        prediction = self.model.predict(hist)[0]
        probability = self.model.predict_proba(hist)[0]
        
        label = 'real' if prediction == 1 else 'spoof'
        score = probability[1] # Probability of being real
        
        return label, score

if __name__ == '__main__':
    # Example usage
    if len(sys.argv) > 1:
        try:
            pipeline = InferencePipeline()
            label, score = pipeline.predict(sys.argv[1])
            print(f"Prediction: {label}, Score: {score:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python inference.py <image_path>")
