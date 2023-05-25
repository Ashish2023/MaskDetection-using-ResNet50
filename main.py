from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model(r"C:\Users\ashis\OneDrive\Desktop\dataset2\Saved models\ResNet50_faceMask.h5")
cap = cv2.VideoCapture(0)  # use 0 for default camera, or specify a camera index if you have multiple cameras

# Define the class labels
class_labels = {'Mask': 0, 'No mask': 1}
labels = dict((v, k) for k, v in class_labels.items())
# your code for starting the process
def start_process():
    pass

# your code for stopping the process
def stop_process():
    # add your code here
    cap.release()
    pass

@app.route('/')
def index():
    return render_template('index.html')

# function to handle the video feed
def gen():
    # add your code to generate the video feed
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize frame to match the input size of the model
            frame = cv2.resize(frame, (224, 224))

            # Preprocess the frame for inference
            x = image.img_to_array(frame)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Make predictions
            preds = model.predict(x)
            pred_classes = np.argmax(preds, axis=1)

            # Map predicted classes to their corresponding class labels
            pred_class_label = labels[pred_classes[0]]
            # Get the percentage of certainty (confidence score) for the prediction
            confidence = preds[0][pred_classes[0]] * 100

            # Change color to red if predicted class is "No Mask"
            if pred_class_label == "No mask":
                color = (0, 0, 255)  # Red color
            else:
                color = (0, 255, 0)  # Green color

            # Draw the predicted class label and confidence score on the frame
            text = f"{pred_class_label}: {confidence:.2f}%"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Encode the frame as a jpeg image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    start_process()
    return "Process started"

@app.route('/stop')
def stop():
    stop_process()
    return "Process stopped"

if __name__ == '__main__':
    app.run(debug=True)

