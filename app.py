from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Predefined KNN values
knn_values = {
    "Mentah": (43.65778653, 0.712169333, 0.587279633),
    "Setengah Matang": (26.4978388, 0.686294933, 0.4887726),
    "Matang": (41.10765907, 0.500538333, 0.35405952),
    "Busuk": (27.67811093, 0.673345, 0.4388876)
}

def allowed_file(filename):
    """Check if a file is an allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_hsv_manual(r, g, b):
    """Calculate HSV values manually from RGB."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    
    s = 0 if mx == 0 else (df / mx)
    v = mx
    
    return h, s, v

def calculate_knn_distance(h, s, v, knn_values):
    """Calculate the nearest neighbor based on HSV values."""
    distances = {}
    for state, (h_ref, s_ref, v_ref) in knn_values.items():
        distances[state] = np.sqrt((h - h_ref) ** 2 + (s - s_ref) ** 2 + (v - v_ref) ** 2)
    return min(distances, key=distances.get)

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and redirect to detection page."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return redirect(url_for('deteksi', filename=filename))
    return redirect(url_for('index'))

@app.route('/extract', methods=['POST'])
def extract_image():
    """Extract RGB and HSV values from the image and display them."""
    filename = request.form['filename']
    if filename and allowed_file(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(filepath)

        # Extract 3x3 pixel sample from the center
        h, w, _ = image.shape
        sample_size = 3
        start_x, start_y = w // 2 - sample_size // 2, h // 2 - sample_size // 2
        sample_image = image[start_y:start_y+sample_size, start_x:start_x+sample_size]

        # Calculate average RGB values
        r_avg = np.mean(sample_image[:, :, 2])
        g_avg = np.mean(sample_image[:, :, 1])
        b_avg = np.mean(sample_image[:, :, 0])

        # Convert RGB to HSV manually
        h_avg, s_avg, v_avg = calculate_hsv_manual(r_avg, g_avg, b_avg)
        
        values = pd.DataFrame({
            "R": [r_avg],
            "G": [g_avg],
            "B": [b_avg],
            "H": [h_avg],
            "S": [s_avg],
            "V": [v_avg]
        })
        
        # Save HSV image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_filename = 'hsv_' + filename
        hsv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], hsv_filename)
        cv2.imwrite(hsv_filepath, hsv_image)

        return render_template('deteksi.html', filename=filename, hsv_filename=hsv_filename, values=values.to_html())
    return redirect(url_for('index'))

@app.route('/detect', methods=['POST'])
def detect():
    """Detect the state of the mangosteen based on HSV values."""
    filename = request.form['filename']
    if filename and allowed_file(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(filepath)
        
        # Extract 3x3 pixel sample from the center
        h, w, _ = image.shape
        sample_size = 3
        start_x, start_y = w // 2 - sample_size // 2, h // 2 - sample_size // 2
        sample_image = image[start_y:start_y+sample_size, start_x:start_x+sample_size]

        # Calculate average RGB values
        r_avg = np.mean(sample_image[:, :, 2])
        g_avg = np.mean(sample_image[:, :, 1])
        b_avg = np.mean(sample_image[:, :, 0])

        # Convert RGB to HSV manually
        h_avg, s_avg, v_avg = calculate_hsv_manual(r_avg, g_avg, b_avg)
        
        # Calculate nearest neighbor (KNN)
        state = calculate_knn_distance(h_avg, s_avg, v_avg, knn_values)
        
        values = pd.DataFrame({
            "R": [r_avg],
            "G": [g_avg],
            "B": [b_avg],
            "H": [h_avg],
            "S": [s_avg],
            "V": [v_avg]
        })
        
        hsv_filename = 'hsv_' + filename

        return render_template('deteksi.html', filename=filename, hsv_filename=hsv_filename, values=values.to_html(), result_text=state)
    return redirect(url_for('index'))

@app.route('/deteksi')
def deteksi():
    """Render the detection results page."""
    filename = request.args.get('filename', '')
    return render_template('deteksi.html', filename=filename)

@app.route('/bantuan')
def bantuan():
    # Logic for the bantuan page
    return render_template('bantuan.html')

@app.route('/display/<filename>')
def display_image(filename):
    """Serve the uploaded image."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
