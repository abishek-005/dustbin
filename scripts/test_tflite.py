import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def test_image(model_path, label_path, image_path):
    # Load the TFLite model
    print(f"Loading model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    labels = load_labels(label_path)

    # Load and preprocess the image
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
        
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image exactly like MobileNetV2 expects (scaling to [-1, 1])
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Run inference
    print("Running inference...")
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get the top prediction
    top_index = np.argmax(predictions)
    confidence = predictions[top_index] * 100

    print("\n--- RESULTS ---")
    print(f"Predicted Category: {labels[top_index]}")
    print(f"Confidence: {confidence:.2f}%")
    
    print("\nAll probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label}: {predictions[i]*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TFLite Model locally")
    parser.add_argument("--model", type=str, default="smart_dustbin_model.tflite", help="Path to the .tflite model")
    parser.add_argument("--labels", type=str, default="labels.txt", help="Path to labels.txt")
    parser.add_argument("--image", type=str, required=True, help="Path to the image you want to test")
    
    args = parser.parse_args()
    test_image(args.model, args.labels, args.image)
