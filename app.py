import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import inference

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

STAGE1_WEIGHTS = os.path.join(app.root_path, 'models', 'stage1_best_model.pth')
STAGE2_WEIGHTS = os.path.join(app.root_path, 'models', 'stage2_best_model.pth')

# ── Load models once at startup, not per-request ──
print("Loading models...")
stage1_model, stage2_model, gradcam_s1, gradcam_s2 = inference.load_models(
    STAGE1_WEIGHTS, STAGE2_WEIGHTS
)
print("Models loaded. Ready to serve.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use PNG or JPG.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = inference.predict_with_gradcam(
            filepath, stage1_model, stage2_model, gradcam_s1, gradcam_s2
        )
    except Exception as e:
        os.remove(filepath)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # ── Clean up the uploaded file after inference ──
    os.remove(filepath)

    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
