"""RetinaCheck backend. Copyright (c) 2026 Vivek Pateriya.

Dataset: Kaggle diabetic-retinopathy 224x224 2019 data.
Third-party libraries retain their own licenses.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

GRADES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
SIDE = 160


def collect_examples(labels_path, image_root):
    """Match labeled image IDs to images in flat or nested directories."""
    image_index = {}
    for path in Path(image_root).rglob('*'):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            if path.stem in image_index:
                raise ValueError(f'Duplicate image ID: {path.stem}')
            image_index[path.stem] = str(path.resolve())
    groups = [[] for _ in GRADES]
    with Path(labels_path).open(newline='', encoding='utf-8-sig') as stream:
        rows = csv.DictReader(stream)
        if not {'id_code', 'diagnosis'}.issubset(rows.fieldnames or []):
            raise ValueError('Labels CSV needs id_code and diagnosis columns')
        seen = set()
        for row in rows:
            identity = row['id_code'].strip()
            grade = int(row['diagnosis'])
            if identity in seen or not 0 <= grade < len(GRADES):
                raise ValueError(f'Duplicate ID or invalid grade: {identity}')
            seen.add(identity)
            if identity not in image_index:
                raise FileNotFoundError(f'Missing image: {identity}')
            groups[grade].append((image_index[identity], grade))
    if any(len(group) < 2 for group in groups):
        raise ValueError('Each grade needs at least two images')
    return groups


def partition_examples(groups, seed=42):
    """Reserve 20 percent of each class without duplicating images."""
    generator = random.Random(seed)
    learning, checking = [], []
    for group in groups:
        ordered = list(group)
        generator.shuffle(ordered)
        holdout = max(1, min(len(ordered)-1, round(len(ordered)*0.2)))
        checking.extend(ordered[:holdout])
        learning.extend(ordered[holdout:])
    generator.shuffle(learning)
    return learning, checking


def decode_retina(path):
    pixels = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                expand_animations=False)
    pixels.set_shape([None, None, 3])
    # Training and prediction share this exact transformation.
    return tf.image.resize(tf.cast(pixels, tf.float32), [SIDE, SIDE]) / 255.0


def assemble_batches(examples, batch_size, learning=False):
    paths, grades = zip(*examples)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(grades)))
    if learning:
        dataset = dataset.shuffle(len(examples), seed=42)
    dataset = dataset.map(lambda path, grade: (decode_retina(path), grade),
                          num_parallel_calls=1)
    return dataset.batch(batch_size).prefetch(1)


def create_retina_network():
    layers = tf.keras.layers
    inputs = layers.Input(shape=(SIDE, SIDE, 3))
    features = layers.RandomFlip('horizontal')(inputs)
    features = layers.RandomRotation(0.04)(features)
    for filters in (12, 24, 48):
        features = layers.Conv2D(filters, 3, strides=2, padding='same',
                                 activation='relu')(features)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dense(64, activation='relu')(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(len(GRADES), activation='softmax')(features)
    network = tf.keras.Model(inputs, outputs, name='vivek_retinacheck')
    network.compile(optimizer=tf.keras.optimizers.Adam(0.0003),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return network


def learn_retina_grades(options):
    tf.keras.utils.set_random_seed(42)
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    groups = collect_examples(options.labels, options.images)
    learning, checking = partition_examples(groups)
    destination = Path(options.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'split.json').write_text(json.dumps(
        {'training': learning, 'validation': checking}, indent=2), encoding='utf-8')
    counts = np.bincount([grade for _, grade in learning], minlength=5)
    weights = {grade: len(learning)/(5*int(count)) for grade, count in enumerate(counts)}
    network = create_retina_network()
    training = assemble_batches(learning, options.batch, learning=True)
    validation = assemble_batches(checking, options.batch)
    print(f'Training images: {len(learning)}, validation images: {len(checking)}')
    network.summary()
    network.fit(training, validation_data=validation, epochs=options.epochs,
                class_weight=weights, callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(str(destination/'best.keras'),
                        monitor='val_loss', save_best_only=True),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                        restore_best_weights=True)])
    network.save(str(destination/'retina.keras'))
    truth, guesses = [], []
    for pixels, labels in validation:
        truth.extend(labels.numpy().tolist())
        guesses.extend(np.argmax(network(pixels, training=False).numpy(), axis=1).tolist())
    matrix = tf.math.confusion_matrix(truth, guesses, num_classes=5).numpy()
    result = {'validation_accuracy': float(np.mean(np.array(truth)==np.array(guesses))),
              'validation_images': len(truth), 'classes': GRADES,
              'confusion_matrix': matrix.tolist()}
    (destination/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Model saved: {destination / "retina.keras"}')


def classify_retina(model_path, image_path):
    network = tf.keras.models.load_model(model_path, compile=False)
    scores = network(decode_retina(image_path)[None, ...], training=False).numpy()[0]
    grade = int(np.argmax(scores))
    return {'grade': grade, 'label': GRADES[grade],
            'probabilities': {label: float(score) for label, score in zip(GRADES, scores)}}


def run_commands():
    parser = argparse.ArgumentParser(description='RetinaCheck by Vivek Pateriya')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--labels', required=True)
    train.add_argument('--images', required=True)
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--batch', type=int, default=2)
    train.add_argument('--output', default='artifacts')
    predict = commands.add_parser('predict')
    predict.add_argument('--model', default='artifacts/retina.keras')
    predict.add_argument('--image', required=True)
    options = parser.parse_args()
    if options.command == 'train':
        if options.batch < 1 or options.epochs < 1:
            parser.error('--batch and --epochs must be positive')
        learn_retina_grades(options)
    else:
        print(json.dumps(classify_retina(options.model, options.image), indent=2))


if __name__ == '__main__':
    run_commands()
