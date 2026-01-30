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
        # 'out_relu' is the last conv layer in standard unpacked MobileNetV2
        return "out_relu"

    def build_gradcam_model(self):
        """
        Creates a new Keras Model that outputs:
        [Target Layer Output, Final Prediction]
        It uses the *existing* graph tensors, avoiding manual reconstruction errors.
        """
        try:
            # 1. Get the actual layer object
            target_layer = self.model.get_layer(self.layerName)
            
            # 2. Get the output tensor of that layer
            conv_output = target_layer.output
            
            # 3. Get the final output tensor of the model
            final_output = self.model.output
            
            # 4. Create the new Multi-Output model sharing the same inputs
            return tf.keras.models.Model(
                inputs=self.model.inputs, 
                outputs=[conv_output, final_output]
            )
            
        except Exception as e:
            print(f"[GradCAM++] Model Tap Failed: {e}")
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

