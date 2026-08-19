import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt

def build_model(num_classes, img_shape=(224, 224, 3)):
    # Data Augmentation pipeline
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ], name="data_augmentation")

    # Load MobileNetV2 pretrained on ImageNet, without the top classification layer
    base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # Create the complete model
    inputs = tf.keras.Input(shape=img_shape)
    # Apply data augmentation
    x = data_augmentation(inputs)
    # MobileNetV2 requires inputs to be in [-1, 1], but image_dataset_from_directory gives [0, 255]
    # We use the built-in preprocess_input
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    dataset_dir = "../dataset_balanced"
    batch_size = 32
    img_size = (224, 224)

    print("Loading dataset...")
    # Load dataset. Keras automatically handles the empty folder as a class if it exists.
    # Note: image_dataset_from_directory might skip empty folders, so we might only get 4 classes.
    # Let's ensure we get exactly the classes we created.
    # If the empty folder is skipped, we can just train on 4 classes for now.
    
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

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    print("Building model...")
    model = build_model(num_classes, img_shape=img_size + (3,))
    model.summary()

    epochs = 10
    print(f"Training model for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # Save Keras Model
    model_save_path = "smart_dustbin_mobilenetv2.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Convert to TFLite
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimize for latency
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
