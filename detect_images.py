import os
#handel file system operation
import cv2
# read and process the uploadingg image 
import numpy as np
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
# for secure file name 
from PIL import Image
# used for resizing
import io

# Load YOLO model files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to perform object detection
def detect_objects(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the result image
    result_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    cv2.imwrite(result_path, img)

    return result_path

# Route for uploading image and processing it
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform object detection
            result_image_path = detect_objects(file_path)

            # Return the result image
            return send_file(result_image_path, mimetype='image/jpeg')
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

