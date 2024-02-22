from flask import Flask, jsonify, render_template, request
import os
import matplotlib.pyplot as plt
import gdown
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from faceswap import simple_face_swap, swap_n_show, swap_n_show_same_img, swap_face_single,fine_face_swap
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import uuid
import base64
app = Flask(__name__)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_output_path = 'inswapper/inswapper_128.onnx'
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Load images
img1_fn = 'images/mrr.jpg'
# img2_fn = 'images/Anushka.jpg'

# Directory where uploaded files will be saved
UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/swap-face', methods=['POST'])
def save_user():
    if 'userImage' in request.files:
        file = request.files['userImage']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    elif request.json and 'base64Image' in request.json:
        # Decode the base64 string
        base64_image = request.json['base64Image']
        image_data = base64.b64decode(base64_image)
        # Create a BytesIO object
        image = Image.open(BytesIO(image_data))
        # Generate a secure random filename
        filename = str(uuid.uuid4()) + '.jpg'
        # Save the image to the filesystem
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        return jsonify({'error': 'No image provided in the request'}), 400

    img2_fn = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Perform the face swap operation
    swapped_image_path =  simple_face_swap(img1_fn, img2_fn, face_app, swapper)
    
    with open(swapped_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return jsonify({'image': encoded_string}), 200


if __name__ == '__main__':
    app.run(debug=True)
