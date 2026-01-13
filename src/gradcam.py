from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.grad_model = None
        
        # Auto-find target if needed
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # Heuristic: MobilenetV2 output has 1280 channels
        for layer in self.model.layers:
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if len(shape) == 4 and shape[-1] == 1280:
                    print(f"[GradCAM++] Found Target Layer: {layer.name}")
                    return layer.name
        # Fallback to user suggestion if not found
        return "mobilenetv2_1.00_224"

    def build_gradcam_model(self):
        """
        Reconstructs a Functional Model to bypass nested model graph issues.
        Maps: New Input -> [Backbone Output, Final Output]
        """
        try:
            # 1. Create Input
            # Note: We assume standard 224x224x3 input for this CNN
            inputs = tf.keras.Input(shape=(224, 224, 3))
            
            # 2. Trace the graph
            # We assume structure: Lambda -> Backbone -> GlobalAvg -> ...
            x = inputs
            
            # Find the index of the backbone
            target_layer = self.model.get_layer(self.layerName)
            backbone_index = self.model.layers.index(target_layer)
            
            # Apply layers UP TO backbone (inclusive) to get conv_output
            # Usually: Input is implicit. Layer 0 is Preprocess (Lambda).
            # But we must act carefully.
            # If we call self.model.layers[0](x), it works.
            
            conv_output = None
            
            # Re-run layers from start to backbone
            for i in range(backbone_index + 1):
                layer = self.model.layers[i]
                # Skip InputLayer if present in list (rare in loaded models list)
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x)
                if i == backbone_index:
                    conv_output = x
            
            # Continue from backbone+1 to end to get predictions
            for i in range(backbone_index + 1, len(self.model.layers)):
                layer = self.model.layers[i]
                x = layer(x)
            
            outputs = x
            
            return tf.keras.Model(inputs=inputs, outputs=[conv_output, outputs])
            
        except Exception as e:
            print(f"[GradCAM++] Model Rebuild Failed: {e}")
            return None

    def compute_heatmap(self, image, eps=1e-8):
        # Lazy Build
        if self.grad_model is None:
            self.grad_model = self.build_gradcam_model()
            if self.grad_model is None:
                return None

        # GRAD-CAM++ LOGIC
        try:
            with tf.GradientTape() as tape:
                inputs = tf.cast(image, tf.float32)
                tape.watch(inputs)
                (conv_outputs, predictions) = self.grad_model(inputs)
                
                # Binary Logic
                score = predictions[0][0]
                if score > 0.5:
                    loss = score
                else:
                    loss = 1.0 - score

            # 1. Gradients (First Order)
            grads = tape.gradient(loss, conv_outputs)
            
            # 2. Grad-CAM++ Higher Order Derivatives
            # grads^2 and grads^3
            grads_power_2 = tf.math.square(grads)
            grads_power_3 = tf.math.pow(grads, 3)

            # 3. Sum over spatial locations (H, W)
            # Axis=(0,1,2) includes Batch? conv_outputs is (B,H,W,C).
            # We want sum over H,W mainly? User snippet: axis=(0,1,2).
            # This sums over Batch, H, W. Result is (C,).
            sum_grads = tf.reduce_sum(conv_outputs * grads_power_2, axis=(0, 1, 2))
            
            # 4. Alpha Calculation
            # alpha = g^2 / (2*g^2 + sum(A*g^3))
            # 2*grads_power_2
            denom = 2 * grads_power_2 + sum_grads + eps
            # Note: broadcasting sum_grads (C,) to (B,H,W,C) works automatically
            
            alpha = grads_power_2 / denom
            
            # 5. Weights
            # weights = sum(alpha * ReLU(grads))
            weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1, 2))
            
            # 6. Heatmap Generation
            # sum(weights * A)
            conv_outputs = conv_outputs[0] # Take first item in batch (H, W, C)
            # weights is (C,).
            
            heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
            
            # 7. ReLU and Normalize
            heatmap = tf.maximum(heatmap, 0)
            
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            
            heatmap = heatmap.numpy()
            
            # Resize
            heatmap = cv2.resize(heatmap, (224, 224))
            
            return heatmap

        except Exception as e:
            print(f"[GradCAM++] Compute Error: {e}")
            import traceback
            traceback.print_exc()
            return None

