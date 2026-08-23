import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(dataset_dir, class_names):
    """Calculate class weights based on the number of images in each folder."""
    y = []
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))])
            y.extend([i] * num_images)
    
    if len(y) == 0:
        return None
        
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(enumerate(class_weights))

def build_base_model(num_classes, img_shape=(224, 224, 3)):
    # Advanced Data Augmentation pipeline
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.3),
        RandomZoom(0.2),
        RandomTranslation(height_factor=0.2, width_factor=0.2),
        RandomContrast(0.2)
    ], name="data_augmentation")

    # Load MobileNetV2 pretrained on ImageNet
    base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model for Phase 1

    # Create the complete model
    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Increased dropout to prevent overfitting
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model, base_model

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "dataset_final")
    batch_size = 32
    img_size = (224, 224)

    print("Loading dataset...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    print(f"Detected classes: {class_names}")
    num_classes = len(class_names)

    # Calculate Class Weights
    print("Calculating Class Weights to handle dataset imbalance...")
    class_weights = get_class_weights(dataset_dir, class_names)
    print(f"Class Weights: {class_weights}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    print("\n--- PHASE 1: Training the Classification Head ---")
    model, base_model = build_base_model(num_classes, img_shape=img_size + (3,))
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    epochs_phase1 = 15
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )

    print("\n--- PHASE 2: Fine-Tuning the Base Model ---")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the top 50
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Recompile with a VERY LOW learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    epochs_phase2 = 15
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_phase2,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save Native Keras Model
    model_save_path = "smart_dustbin_mobilenetv2.keras"
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # Convert to TFLite
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_save_path = "smart_dustbin_model.tflite"
    with open(tflite_save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model successfully saved to {tflite_save_path}")

    # Write a label map file
    with open("labels.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Labels saved to labels.txt")

if __name__ == "__main__":
    main()
