"""Local RetinaCheck interface. Copyright (c) 2026 Vivek Pateriya."""
import tempfile
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
MODEL_PATH = Path(__file__).parent / 'artifacts' / 'retina.keras'


@app.get('/')
def show_workspace():
    return render_template('index.html')


@app.get('/about')
def show_project_paper():
    metrics = None
    try:
        metrics = json.loads((MODEL_PATH.parent / 'metrics.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    return render_template('about.html', metrics=metrics)


@app.post('/predict')
def analyse_upload():
    upload = request.files.get('image')
    if upload is None or not upload.filename:
        return jsonify(error='Choose a PNG or JPEG image first.'), 400
    try:
        with Image.open(upload.stream) as picture:
            if picture.format not in {'PNG', 'JPEG'}:
                return jsonify(error='Please use a PNG or JPEG image.'), 400
            picture.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        return jsonify(error='This file could not be read as an image.'), 400
    if not MODEL_PATH.is_file():
        return jsonify(error='The new model is not ready yet. Train RetinaCheck first.'), 503
    upload.stream.seek(0)
    with tempfile.TemporaryDirectory(prefix='retinacheck-') as temporary:
        path = Path(temporary) / 'upload'
        upload.save(path)
        from retina import classify_retina
        result = classify_retina(str(MODEL_PATH), str(path))
    return jsonify(result)


@app.errorhandler(413)
def oversized_upload(_error):
    return jsonify(error='Choose an image smaller than 10 MB.'), 413


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
