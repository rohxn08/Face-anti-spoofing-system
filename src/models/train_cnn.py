import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Lambda
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_cnn_model():
    # --- CONFIGURATION ---
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)
    EPOCHS = 15
    
    # Paths (Relative to script location)
    # script is in src/models/, so we go up two levels to get to project root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'Detectedface')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Project Root: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    # --- DATA PREPARATION ---
    # 1. Define Generators
    # Training: Augmented (Safe for cropped faces)
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=10,       # Reduced rotation to keep corners safe
        width_shift_range=0.1,   # Reduced shift to keep features in frame
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,    # Safe and effective
        fill_mode='nearest'
        # Note: No rescale=1./255 because MobileNetV2 preprocess_input handles it
    )

    # Validation: No Augmentation
    val_datagen = ImageDataGenerator(validation_split=0.2)
    
    # 2. Load Data
    print("\nLoading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    print("Loading Validation Data...")
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    print(f"Classes: {train_generator.class_indices}")

    # --- MODEL ARCHITECTURE ---
    print("\nBuilding Model...")
    model = Sequential()
    model.add(Input(shape=(128, 128, 3)))
    
    # 1. Preprocessing Layer (MobileNetV2 expects [-1, 1])
    model.add(Lambda(preprocess_input))
    
    # 2. Base Model (Transfer Learning)
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    base_model.trainable = False # Freeze base layers
    model.add(base_model)
    
    # 3. Head (Classifier)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # --- COMPILATION ---
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # --- CALLBACKS ---
    model_save_path = os.path.join(MODELS_DIR, 'face_antispoofing_model.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path, 
            monitor='val_loss', 
            save_best_only=True,
            verbose=1
        )
    ]

    # --- TRAINING ---
    print("\nStarting Training...")
    try:
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        print("\nTraining Completed.")
        
        # --- PLOTTING HISTORY ---
        plot_history(history, MODELS_DIR)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")

def plot_history(history, save_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    train_cnn_model()
