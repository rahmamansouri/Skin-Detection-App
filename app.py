from flask import Flask, send_file
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

@app.route('/get_result_image')
def get_result_image():
    # Path to the image you want to test
    image_path = "images/birthmark.jpg"

    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to (224, 224) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape
    image_for_prediction = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_names[index].rstrip()  # Remove trailing whitespace

    # Create a larger result image with a title
    result_image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background

    # Calculate the position to center the image
    center_y = (result_image.shape[0] - image.shape[0]) // 2
    center_x = (result_image.shape[1] - image.shape[1]) // 2

    # Insert the image into the frame
    result_image[center_y:center_y + image.shape[0], center_x:center_x + image.shape[1], :] = image
    # Draw title
    cv2.putText(result_image, "Skin Detection", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw bounding box on the result image
    cv2.rectangle(result_image, (center_x, center_y), (center_x + image.shape[1], center_y + image.shape[0]), (0, 0, 0), 2)  # Black frame

    # Display predicted class and confidence at the bottom
    confidence_score = prediction[0][index]  # Confidence score is now at index
    cv2.putText(result_image, f"Class: {class_name[2:]}", (110, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(result_image, f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%", (110, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Save the result image temporarily
    cv2.imwrite("result_image.jpg", result_image)

    # Send the saved image file
    return send_file("result_image.jpg", mimetype='image/jpeg') #This line sends the saved image file to the client (in this case, samsung A32 mobile device). The send_file function is a Flask function that sends the contents of a file to the client. 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #start the development server

