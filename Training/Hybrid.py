"""Read main() from top to bottom: load -> build -> compile -> fit -> evaluate."""
from pathlib import Path
import json
import numpy as np
from dataset import keras, configure_training, load_data, augment_images, CLASSES, MEAN, STD
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# These imports only reuse architecture definitions; they do NOT start training.
from CNN import cnn_features
from ViT import vit_features


def build_model():
    inputs = keras.Input(shape=(224, 224, 3), name='clahe_pixels')
    images = augment_images(inputs)  # Both branches see the SAME augmented image.
    cnn = cnn_features(images)  # 256 features
    vit = vit_features(images)  # 256 features
    merged = keras.layers.Concatenate(name='hybrid_features')([cnn, vit])  # 512 features
    x = keras.layers.Dense(256, activation='relu')(merged)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(2, activation='softmax', dtype='float32', name='classifier')(x)
    return keras.Model(inputs, outputs, name='Hybrid')


def main():
    # 1. Settings. Change these here; no command-line options are needed.
    epochs = 30
    batch_size = 16
    accumulation_steps = 2  # 16 x 2 = 32 images per optimizer update.
    output = Path(__file__).resolve().parent / 'Hybrid'
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'console.log' for p in output.iterdir()):
        raise FileExistsError(f'{output} already contains results. Move them before a new run.')
    configure_training()

    # 2. Read the existing splits; do not redo preprocessing or splitting.
    train_data, val_data, test_data, labels = load_data(batch_size, accumulation_steps)

    # 3. Build and compile the classifier.
    model = build_model()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01,
                                        gradient_accumulation_steps=accumulation_steps,
                                        global_clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=False,  # Avoid an extra XLA compilation requirement on this laptop.
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(output / 'best_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.CSVLogger(output / 'training_history.csv'),
        keras.callbacks.TerminateOnNaN(),
    ]
    (output / 'settings.json').write_text(json.dumps({
        'model': 'Hybrid', 'epochs': epochs, 'batch_size': batch_size,
        'accumulation_steps': accumulation_steps, 'classes': CLASSES,
        'input': 'CLAHE pixels, float32 HWC (224,224,3), range 0..255',
        'vit_preset': 'vit_base_patch16_224_imagenet21k' if 'Hybrid' != 'CNN' else None,
        'vit_backbone_frozen': 'Hybrid' != 'CNN',
    }, indent=2))

    # 4. Keras performs training and validation for us, just like your earlier project.
    history = model.fit(train_data, validation_data=val_data, epochs=epochs,
                        callbacks=callbacks, verbose=2, shuffle=False)
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise RuntimeError('Training produced non-finite values. Check console.log.')
    model.load_weights(output / 'best_model.keras')  # Select by validation loss only.
    print('Training finished. Restored the best validation checkpoint.')

    # 5. Final validation and Shenzhen evaluation. Test data never chooses the model.
    results = {'best_epoch': int(np.argmin(history.history['val_loss']) + 1)}
    reports = []
    for split, data in [('val', val_data), ('test', test_data)]:
        probabilities = model.predict(data, verbose=0)
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Non-finite predictions')
        predicted = np.argmax(probabilities, axis=1)
        actual = labels[split]
        report = classification_report(actual, predicted, labels=[0, 1],
                                       target_names=CLASSES, zero_division=0)
        metrics = classification_report(actual, predicted, labels=[0, 1],
                                        target_names=CLASSES, zero_division=0, output_dict=True)
        accuracy = float(np.mean(actual == predicted))
        auc = float(roc_auc_score(actual, probabilities[:, 1]))
        matrix = confusion_matrix(actual, predicted, labels=[0, 1]).tolist()
        results[split] = {'accuracy': accuracy, 'auc': auc, 'classification_report': metrics,
                          'confusion_matrix': matrix}
        text = f'{split.upper()} accuracy: {accuracy:.2%} | AUC: {auc:.4f}\n{report}\nConfusion matrix: {matrix}'
        print(text)
        reports.append(text)
    gap = 100 * (results['val']['accuracy'] - results['test']['accuracy'])
    results['accuracy_gap_percentage_points'] = gap
    summary = (f"Within-dataset accuracy: {results['val']['accuracy']:.2%}\n"
               f"Cross-dataset accuracy: {results['test']['accuracy']:.2%}\n"
               f'Accuracy gap: {gap:.2f} percentage points')
    print(summary)
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    (output / 'evaluation.txt').write_text('\n\n'.join(reports + [summary]))
    print('Saved model, history and evaluation in:', output)


if __name__ == '__main__':
    main()
