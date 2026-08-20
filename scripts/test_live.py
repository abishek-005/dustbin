import cv2
import tensorflow as tf
import numpy as np
import argparse

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main(model_path, label_path):
    print(f"Loading model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    labels = load_labels(label_path)

    # Initialize the webcam
    print("Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # 1. Preprocess the frame for the model
        # OpenCV captures in BGR, but our model was trained on RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 as expected by MobileNetV2
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        
        # Convert to float32 and add batch dimension
        img_array = np.array(resized_frame, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply MobileNetV2 preprocessing (scales pixels to [-1, 1])
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # 2. Run Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # 3. Get the top prediction
        top_index = np.argmax(predictions)
        confidence = predictions[top_index] * 100
        predicted_label = labels[top_index]

        # 4. Display the result on the video feed
        text = f"{predicted_label}: {confidence:.2f}%"
        
        # Change color based on confidence (Green if > 75%, Red otherwise)
        color = (0, 255, 0) if confidence > 75 else (0, 0, 255)
        
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the live video window
        cv2.imshow("Smart Dustbin - Live Testing", frame)

        # Press 'q' on the keyboard to exit the live view
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TFLite Model Live via Webcam")
    parser.add_argument("--model", type=str, default="smart_dustbin_model.tflite", help="Path to the .tflite model")
    parser.add_argument("--labels", type=str, default="labels.txt", help="Path to labels.txt")
    
    args = parser.parse_args()
    main(args.model, args.labels)
