import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    RandomFlip, RandomRotation, RandomZoom,
    RandomTranslation, RandomContrast, RandomBrightness
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(dataset_dir, class_names):
    """Calculate class weights based on the number of images in each folder."""
    y = []
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            num_images = len([f for f in os.listdir(class_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))])
            y.extend([i] * num_images)

    if len(y) == 0:
        return None

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(enumerate(class_weights))


def build_model(num_classes, img_shape=(224, 224, 3)):
    """Build MobileNetV2 model with strong data augmentation."""

    # Aggressive Data Augmentation pipeline — forces the model to learn
    # the actual object shape/texture, not just background colors.
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.3),
        RandomZoom((-0.2, 0.2)),
        RandomTranslation(height_factor=0.2, width_factor=0.2),
        RandomContrast(0.3),
        RandomBrightness(factor=0.3),
    ], name="data_augmentation")

    # Load MobileNetV2 pretrained on ImageNet
    base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze for Phase 1

    # Build the model
    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)  # Extra dense layer for better feature separation
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model, base_model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "dataset_final")
    batch_size = 32
    img_size = (224, 224)

    # Verify dataset exists
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset directory not found at: {os.path.abspath(dataset_dir)}")
        print("Make sure 'dataset_final' folder exists next to the 'scripts' folder.")
        return

    print(f"Using dataset: {os.path.abspath(dataset_dir)}")
    print("Loading dataset...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    class_names = train_dataset.class_names
    print(f"\nDetected classes: {class_names}")
    num_classes = len(class_names)

    # Count images per class
    for name in class_names:
        class_path = os.path.join(dataset_dir, name)
        count = len(os.listdir(class_path))
        print(f"  {name}: {count} images")

    # Calculate Class Weights
    print("\nCalculating class weights...")
    class_weights = get_class_weights(dataset_dir, class_names)
    print(f"Class weights: {class_weights}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    model, base_model = build_model(num_classes, img_shape=img_size + (3,))

    # Use label smoothing — prevents overconfident wrong predictions.
    # Instead of the model learning "100% paper, 0% everything else",
    # it learns "95% paper, 1.25% each other class". This makes it
    # much better at generalizing to new images it has never seen.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=7, restore_best_weights=True, mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
    checkpoint = ModelCheckpoint(
        'best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )

    callbacks = [early_stopping, reduce_lr, checkpoint]

    # =========================================================================
    # PHASE 1: Train only the classification head (base model frozen)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training the Classification Head (15 epochs)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # =========================================================================
    # PHASE 2: Unfreeze top 50 layers, fine-tune with low LR
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-Tuning Top 50 Layers (20 epochs)")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # =========================================================================
    # PHASE 3: Unfreeze top 100 layers, ultra-low LR for deep fine-tuning
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Deep Fine-Tuning Top 100 Layers (15 epochs)")
    print("=" * 60)

    for layer in base_model.layers[:-100]:
        layer.trainable = False
    for layer in base_model.layers[-100:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # =========================================================================
    # SAVE: Load the best model from checkpoint and export
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Load the absolute best model across all 3 phases
    best_model = tf.keras.models.load_model('best_model.keras')

    # Evaluate on validation set
    print("\nFinal validation accuracy:")
    best_model.evaluate(val_dataset)

    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_save_path = "smart_dustbin_model.tflite"
    with open(tflite_save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_save_path}")

    # Save labels
    with open("labels.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Labels saved to labels.txt")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Files created: {tflite_save_path}, labels.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
