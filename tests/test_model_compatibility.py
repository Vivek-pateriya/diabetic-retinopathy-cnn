"""Smoke tests for the bundled model on supported Python/TensorFlow versions."""
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from retina import classify_retina, create_retina_network


class ModelCompatibilityTests(unittest.TestCase):
    def test_bundled_model_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'input.png'
            Image.new('RGB', (224, 224), (60, 30, 20)).save(image)
            result = classify_retina(ROOT / 'artifacts' / 'retina.keras', image)
        scores = list(result['probabilities'].values())
        self.assertEqual(len(scores), 5)
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(sum(scores), 1.0, places=5)
        self.assertIn(result['grade'], range(5))

    def test_batch_training_and_serialization(self):
        # New temporary network only; never overwrite the bundled model.
        model = create_retina_network()
        inputs = np.zeros((2, 160, 160, 3), dtype='float32')
        loss = model.train_on_batch(inputs, np.array([0, 1]))
        self.assertTrue(np.isfinite(loss).all())
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory) / 'smoke.keras')

    def test_flask_upload(self):
        import app as web
        from unittest.mock import patch
        picture = io.BytesIO()
        Image.new('RGB', (160, 160)).save(picture, format='PNG')
        picture.seek(0)
        with patch.object(web, 'MODEL_PATH', ROOT / 'artifacts' / 'retina.keras'):
            client = app.test_client()
            self.assertEqual(client.get('/').status_code, 200)
            response = client.post('/predict', data={'image': (picture, 'test.png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()['probabilities']), 5)


if __name__ == '__main__':
    unittest.main()
