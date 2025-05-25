import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSION = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


image_size = (200, 200)
Model_path = 'NotPets_model.h5'
Class_names = ['Cheetah','Hyena']

model = None
model = load_model(Model_path)

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Upload', methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                img = Image.open(filepath).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array/ 255.0
                predictions = model.predict(img_array)[0][0]
                if predictions > 0.5:
                    prediction_class = 1
                else:
                    prediction_class = 0

                result_label = f"{Class_names[prediction_class]}"
                sig = f"sigmoid_result: ({predictions:.3f})"
                return render_template('result.html',
                                        image_url=url_for('static', filename=f'uploads/{filename}'),
                                        classification_result=result_label, sig = sig)
        
            except Exception as e:
                return render_template('error.html', message = f'Error during Classification Process: {e}')
    
        else:
            return render_template('error.html', message="Invalid file type. Allowed: png, jpg, jpeg, gif")
    return render_template('Upload.html')

if __name__ == '__main__':
    app.run(debug = True)