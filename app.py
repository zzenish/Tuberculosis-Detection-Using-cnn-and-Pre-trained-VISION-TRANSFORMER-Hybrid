"""University project demo. Run with: bash startUI.sh"""
from pathlib import Path
import io
import json
import threading

import cv2
import numpy as np
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'Training/Hybrid/best_model.keras'
CLASSES = ['Normal', 'Tuberculosis']


def prepare_image(contents):
    """Match preprocessing steps 02–05. Normalization is inside the saved model."""
    # Check dimensions before OpenCV allocates the decoded image.
    with Image.open(io.BytesIO(contents)) as image:
        if image.format not in ('PNG', 'JPEG'):
            raise ValueError('Please upload a PNG or JPEG image.')
        if image.width * image.height > 25_000_000:
            raise ValueError('Please use an image smaller than 25 megapixels.')
        if image.mode.startswith('I') or image.mode == 'F':
            raise ValueError('Please export an 8-bit PNG or JPEG, matching the training images.')
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Please upload a single image.')
        image.verify()
    gray = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('This image could not be read. Try another PNG or JPEG.')
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    pixels = np.repeat(contrast[..., None], 3, axis=-1).astype(np.float32)
    return gray, contrast, pixels[None, ...]


@st.cache_resource(show_spinner=False)
def load_model():
    # Reuse the project's CUDA library fix. No training functions are called.
    from Training.dataset import tf, keras
    import keras_hub  # Registers ViT layers when reading the saved model.
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = keras.models.load_model(MODEL, compile=False)
    return model, threading.Lock()


def predict(pixels):
    model, lock = load_model()
    with lock:
        scores = np.asarray(model(pixels, training=False))[0].astype(float)
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ValueError('The model returned an invalid result.')
    return scores


def main():
    st.set_page_config(page_title='Chest X-ray | TB Project', page_icon='🩻', layout='wide')
    st.markdown('''<style>
    .block-container {max-width:1100px; padding-top:2.5rem;}
    h1 {letter-spacing:-0.035em;}
    [data-testid="stMetric"] {background:#f1f5f9; border-radius:12px; padding:16px; color:#172b4d;}
    </style>''', unsafe_allow_html=True)
    st.caption('FINAL YEAR PROJECT · COMPUTER ENGINEERING')
    st.title('Tuberculosis classification from chest X-rays')
    st.write('Upload a chest X-ray to explore the prediction of our CNN–Vision Transformer hybrid model.')
    st.info('Academic demonstration only. A prediction is not a diagnosis. The model cannot verify that an uploaded image is a chest X-ray.')
    detection, evaluation, about = st.tabs(['Try an X-ray', 'Evaluation results', 'About the project'])
    with detection:
        st.subheader('Upload an image')
        st.caption('PNG or JPEG · Up to 10 MB · Use an anonymised image. Uploads are not saved by this app.')
        upload = st.file_uploader('Chest X-ray', type=['png', 'jpg', 'jpeg'])
        if upload is not None:
            if upload.size > 10 * 1024 * 1024:
                st.error('Please use an image smaller than 10 MB.')
            else:
                try:
                    gray, contrast, pixels = prepare_image(upload.getvalue())
                except Exception as exc:
                    st.error(f'Unable to prepare the image: {exc}')
                else:
                    left, right = st.columns(2)
                    with left:
                        st.image(gray, caption='Uploaded image · grayscale', width='stretch', clamp=True)
                    with right:
                        st.image(contrast, caption='Model input · resized and contrast adjusted', width='stretch', clamp=True)
                    if st.button('Analyse X-ray', type='primary'):
                        if not MODEL.is_file():
                            st.error('Hybrid checkpoint is missing. Complete Hybrid training first.')
                        else:
                            try:
                                with st.spinner('Loading the model and analysing the image…'):
                                    scores = predict(pixels)
                                label = CLASSES[int(np.argmax(scores))]
                                st.subheader(f'Model prediction: {label}')
                                a, b = st.columns(2)
                                a.metric('Normal score', f'{scores[0]:.1%}')
                                b.metric('Tuberculosis score', f'{scores[1]:.1%}')
                                st.caption('These are model scores, not calibrated probabilities of disease. The class with the higher score is selected, as in the reported evaluation.')
                            except Exception as exc:
                                st.error('Prediction could not be completed. Check the environment and model file.')
                                with st.expander('Error details'):
                                    st.code(str(exc))
    with evaluation:
        st.subheader('Performance on the separate Shenzhen test set')
        st.write('Training and validation use the pooled training sources. Shenzhen is excluded from training. Results below are read from the completed runs.')
        rows = []
        for name in ['CNN', 'ViT', 'Hybrid']:
            path = ROOT / 'Training' / name / 'results.json'
            if path.is_file():
                result = json.loads(path.read_text())
                test = result['test']
                rows.append({'Model': name, 'Validation accuracy': f"{result['val']['accuracy']:.2%}",
                             'Test accuracy': f"{test['accuracy']:.2%}",
                             'TB recall': f"{test['classification_report']['Tuberculosis']['recall']:.2%}",
                             'Specificity': f"{test['classification_report']['Normal']['recall']:.2%}",
                             'TB F1': f"{test['classification_report']['Tuberculosis']['f1-score']:.2%}",
                             'AUC': f"{test['auc']:.4f}"})
        if rows:
            st.dataframe(rows, hide_index=True, width='stretch')
        else:
            st.info('Evaluation results will appear after training completes.')
        path = ROOT / 'Training/Hybrid/results.json'
        if path.is_file():
            matrix = json.loads(path.read_text())['test']['confusion_matrix']
            st.markdown('**Hybrid: correct predictions and errors**')
            st.table([{'Actual class': 'Normal', 'Predicted Normal': matrix[0][0], 'Predicted TB': matrix[0][1]},
                      {'Actual class': 'Tuberculosis', 'Predicted Normal': matrix[1][0], 'Predicted TB': matrix[1][1]}])
        st.caption('Recall measures TB cases detected. Specificity measures Normal cases correctly identified. Shenzhen results have been reviewed across multiple experiments; they are not a newly untouched benchmark.')
    with about:
        st.subheader('How the prediction works')
        st.write('Image → grayscale → resize to 224 × 224 → CLAHE contrast adjustment → three channels → CNN and ViT → feature fusion → two class scores.')
        st.write('The CNN produces 256 features. The pretrained ViT classification token is projected to 256 features. The Hybrid combines them into 512 features and learns a classification head. The ViT backbone is frozen in this implementation.')
        st.write('The saved model includes the normalization required by each branch. The interface supplies CLAHE pixels in the range 0–255. Random augmentation is disabled during prediction.')
        st.markdown('**Classes:** Normal and dataset-labelled Tuberculosis. The labels do not establish confirmed active TB.')
        st.markdown('**Limitations:** Performance varies between dataset sources. The model can miss TB and produce false alarms. It was not validated for clinical use or for arbitrary photographs, CT scans or DICOM uploads.')
        st.caption('Model file: Training/Hybrid/best_model.keras · This interface does not retrain or modify the model.')


if __name__ == '__main__':
    main()
