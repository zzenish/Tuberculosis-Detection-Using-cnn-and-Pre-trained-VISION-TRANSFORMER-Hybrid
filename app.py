"""One-file UI: image preparation, prediction, Grad-CAM and Streamlit pages.
Launch with ./startUI.sh. Training code and model files are separate.
"""
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
import os, io, json, sys, hashlib, subprocess, time, warnings
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent

# Prediction runs in a short-lived copy of this same file. This releases model
# memory after each request and avoids filling the laptop's 4 GB GPU.
def prediction_worker():
    name = sys.argv[2]
    if name not in ['CNN', 'ViT', 'Hybrid']:
        raise ValueError('Unknown model')
    pixels = np.load(io.BytesIO(sys.stdin.buffer.read()), allow_pickle=False)
    if pixels.shape != (1,224,224,3) or not np.isfinite(pixels).all():
        raise ValueError('Invalid model input')
    # Keep library output separate from the JSON response.
    with __import__('contextlib').redirect_stdout(sys.stderr):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        import keras
        import keras_hub
        path = Path(__file__).resolve().parent/'Training'/name/'best_model.keras'
        model = keras.models.load_model(path, compile=False)
        scores = np.asarray(model(pixels, training=False))[0].astype(float)
        heatmap = None
        if '--explain' in sys.argv and name != 'ViT':
            # Final CNN convolution; explain the predicted-class score.
            probe = keras.Model(model.inputs, [model.get_layer('cnn_conv_256').output, model.output])
            inputs = tf.convert_to_tensor(pixels)
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                activations, probabilities = probe(inputs, training=False)
                score = probabilities[:, int(np.argmax(scores))]
            gradients = tape.gradient(score, activations)
            if gradients is None:
                raise RuntimeError('CNN explanation gradients are unavailable.')
            weights = tf.reduce_mean(tf.cast(gradients, tf.float32), axis=(1,2))
            cam = tf.reduce_sum(tf.cast(activations,tf.float32)*weights[:,None,None,:],axis=-1)
            cam = tf.nn.relu(cam)[0]
            cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam))
            cam = tf.image.resize(cam[None,...,None], (224,224))[0,:,:,0]
            heatmap = cam.numpy().tolist()
    if scores.shape != (2,) or not np.isfinite(scores).all() or not np.isclose(scores.sum(),1,atol=.001):
        raise ValueError('Invalid prediction')
    print(json.dumps({'prediction':['Normal','Tuberculosis'][int(np.argmax(scores))],
                      'normal_score':float(scores[0]), 'tb_score':float(scores[1]), 'heatmap':heatmap}))


# Image validation and the interface-to-model connection.
LABELS=['Normal','Dataset-labeled TB']

def decode_upload(content):
    if not content or len(content)>20*1024*1024:
        raise ValueError('Choose a nonempty PNG or JPEG file smaller than 20 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as im:
                if im.format not in {'PNG','JPEG'}: raise ValueError('Only PNG and JPEG are supported.')
                if im.width*im.height>40_000_000: raise ValueError('Image exceeds the 40 megapixel limit.')
                if im.mode not in {'L','RGB','RGBA','P'}: raise ValueError('Use an 8-bit grayscale or RGB PNG/JPEG image.')
                im.verify()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise ValueError('This file could not be decoded as a valid PNG/JPEG image.') from exc
    gray=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError('This file could not be decoded as a valid image.')
    return gray

class Predictor:
    def __init__(self,path):
        self.path=Path(path)
        self.name=self.path.parent.name
        digest=hashlib.sha256()
        with self.path.open('rb') as file:
            for chunk in iter(lambda:file.read(1024*1024),b''): digest.update(chunk)
        result=json.loads((self.path.parent/'results.json').read_text())
        self.metadata={'selected_epoch':result['best_epoch'],'source_checkpoint_sha256':digest.hexdigest()}

    def predict(self,content,explain=True):
        started=time.monotonic()
        gray=decode_upload(content)
        resized=cv2.resize(gray,(224,224),interpolation=cv2.INTER_AREA)
        contrast=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(resized)
        processed=np.repeat(contrast[...,None],3,axis=-1)
        stream=BytesIO();np.save(stream,processed[None].astype(np.float32),allow_pickle=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--predict-worker',self.name]
        if explain and self.name!='ViT':command.append('--explain')
        try:
            response=subprocess.run(command,input=stream.getvalue(),capture_output=True,timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Prediction took longer than three minutes. Try again without the explanation.') from exc
        if response.returncode:
            error=response.stderr.decode(errors='replace')[-2000:]
            print(error,flush=True)
            raise RuntimeError(f'Prediction failed: {error}')
        output=json.loads(response.stdout)
        index=int(output['tb_score']>output['normal_score'])
        heatmap=np.asarray(output['heatmap'],dtype=np.float32) if output.get('heatmap') is not None else None
        overlay=None
        if heatmap is not None:
            color=cv2.cvtColor(cv2.applyColorMap((heatmap*255).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
            overlay=cv2.addWeighted(processed,.6,color,.4,0)
        return dict(predicted_class=LABELS[index],class_index=index,normal_score=output['normal_score'],
                    tb_score=output['tb_score'],seconds=time.monotonic()-started,processed=processed,
                    overlay=overlay,heatmap_informative=bool(np.ptp(heatmap)>1e-6) if heatmap is not None else None)


def main():
    import streamlit as st
    st.set_page_config(page_title='Chest X-ray Assessment',page_icon='🩻',layout='wide')
    st.markdown('''<style>
    .block-container{max-width:1180px;padding-top:2.5rem;padding-bottom:3rem}
    h1{font-size:2.6rem!important;letter-spacing:-.045em} h2{letter-spacing:-.025em}
    .eyebrow{color:#176b72;font-size:.8rem;font-weight:700;letter-spacing:.14em;margin-bottom:.6rem}
    .intro{color:#536574;font-size:1.1rem;max-width:760px;margin-bottom:1.8rem}
    [data-testid="stMetric"]{background:white;border:1px solid #dfe7eb;border-radius:10px;padding:18px}
    [data-testid="stSidebar"]{border-right:1px solid #dfe7eb}
    </style>''',unsafe_allow_html=True)
    with st.sidebar:
     st.markdown('### Chest X-ray study')
     st.caption('Final year major project')
     page=st.radio('Navigation',['X-ray assessment','Model results','Methodology'],label_visibility='collapsed')
     st.divider()
     st.markdown('**CNN + Vision Transformer**')
     st.caption('Binary classification of Normal and dataset-labeled Tuberculosis.')
     st.caption('Research demonstration. Not for clinical decisions.')

    @st.cache_resource(show_spinner=False)
    def predictor(name,modified):return Predictor(ROOT/'Training'/{'cnn':'CNN','vit':'ViT','hybrid':'Hybrid'}[name]/'best_model.keras')

    def model_results():
     import csv
     def metrics(value):
      tb=value['classification_report']['Tuberculosis']
      return dict(accuracy=value['accuracy'],precision=tb['precision'],recall=tb['recall'],
                  f1=tb['f1-score'],auc=value['auc'],confusion_matrix=value['confusion_matrix'])
     result=[]
     for name,folder in [('cnn','CNN'),('vit','ViT'),('hybrid','Hybrid')]:
      p=ROOT/'Training'/folder
      if not (p/'results.json').exists():continue
      data=json.loads((p/'results.json').read_text())
      with (p/'training_history.csv').open() as file:
       hist=[dict(epoch=int(r['epoch'])+1,train_loss=float(r['loss']),
                  validation={'loss':float(r['val_loss']),'accuracy':float(r['val_accuracy'])}) for r in csv.DictReader(file)]
      best={'epoch':data['best_epoch'],'validation':metrics(data['val'])}
      test={'test':metrics(data['test']),'accuracy_gap_percentage_points':data['accuracy_gap_percentage_points']}
      result.append((name,best,hist,test))
     return result

    if page=='X-ray assessment':
     st.markdown('<div class="eyebrow">CHEST X-RAY ASSESSMENT</div>',unsafe_allow_html=True)
     st.title('A closer look at the X-ray')
     st.markdown('<div class="intro">Compare the model’s prediction with the input image and examine the regions contributing to its decision.</div>',unsafe_allow_html=True)
     left,right=st.columns([1,1.15],gap='large')
     with left:
      st.subheader('1. Select an image')
      model=st.selectbox('Model',['hybrid','cnn','vit'],format_func=lambda x:{'hybrid':'CNN + ViT hybrid','cnn':'Custom CNN','vit':'Pretrained ViT'}[x])
      upload=st.file_uploader('Chest X-ray',type=['png','jpg','jpeg'],help='8-bit PNG or JPEG, up to 20 MB. DICOM is not supported.')
      st.caption('Use a chest X-ray. The system does not verify anatomy or image suitability. Images are processed in memory and are not saved by this app.')
      explain=st.checkbox('Show CNN explanation',value=True,disabled=model=='vit')
      if model=='vit':st.caption('CNN Grad-CAM is available for the CNN and hybrid models only.')
      content=upload.getvalue() if upload else None
      bundle=ROOT/'Training'/{'cnn':'CNN','vit':'ViT','hybrid':'Hybrid'}[model]/'best_model.keras'
      signature=(hashlib.sha256(content).hexdigest() if content else None,model,explain,bundle.stat().st_mtime_ns if bundle.exists() else None)
      if st.session_state.get('signature')!=signature:
       st.session_state.pop('result',None);st.session_state['signature']=signature
      valid=False
      if content:
       try:
        gray=decode_upload(content);st.image(gray,caption='Uploaded image',width='stretch');valid=True
       except ValueError as e:st.error(str(e))
      if not bundle.exists():st.error('Model unavailable. Complete training for the selected model first.')
      if st.button('Assess image',type='primary',width='stretch',disabled=not valid or not bundle.exists()):
       try:
        with st.spinner('Processing image… Model loading may take a moment.'):
         engine=predictor(model,signature[-1]);result=engine.predict(content,explain=explain and model!='vit')
         result['report']={'image':upload.name,'image_sha256':signature[0],'model':model,'selected_epoch':engine.metadata['selected_epoch'],'checkpoint_sha256':engine.metadata['source_checkpoint_sha256'],'prediction':result['predicted_class'],'normal_score':result['normal_score'],'tb_score':result['tb_score'],'created_utc':datetime.now(timezone.utc).isoformat(),'note':'Research classification of source labels; not a diagnosis. Scores are not calibrated disease probabilities.'}
         st.session_state['result']=result
       except (ValueError,RuntimeError,OSError,KeyError) as e:st.error(f'Unable to process this image: {e}')
     with right:
      st.subheader('2. Review the result')
      result=st.session_state.get('result')
      if result is None:
       st.info('Select an X-ray and choose Assess image. The prediction and processed image will appear here.')
       st.markdown('**What the model does**\n\nThe CNN learns image patterns. The transformer represents relationships between image patches. The hybrid combines their features before classification.')
      else:
       st.markdown(f'### {result["predicted_class"]}')
       a,b=st.columns(2);a.metric('Normal score',f'{result["normal_score"]:.1%}');b.metric('TB score',f'{result["tb_score"]:.1%}')
       st.caption('These are model scores, not a person’s probability of having active tuberculosis.')
       a,b=st.columns(2);a.image(result['processed'],caption='Preprocessed input · 224 × 224',width='stretch')
       if result['overlay'] is not None:
        b.image(result['overlay'],caption='CNN contribution to predicted class',width='stretch')
        st.caption('Grad-CAM highlights contributions from the CNN branch. It is not a lesion outline or a transformer attention map.')
        if not result['heatmap_informative']:st.info('This explanation is flat; it does not identify a localized region.')
       st.download_button('Download assessment record',json.dumps(result['report'],indent=2),'assessment.json','application/json')
     st.divider();st.caption('Tuberculosis Detection from Chest X-rays using CNN and Pre-trained Vision Transformer · Academic research prototype')
    elif page=='Model results':
     st.markdown('<div class="eyebrow">EXPERIMENT RESULTS</div>',unsafe_allow_html=True)
     st.title('How the models performed')
     st.markdown('Validation uses held-out images from the development sources. Testing uses the separate Shenzhen collection.')
     records=model_results();table=[]
     for name,best,hist,test in records:
      table.append({'Model':name.upper(),'Best epoch':best['epoch'],'Epochs run':len(hist),'Validation accuracy':f'{best["validation"]["accuracy"]:.2%}','Shenzhen accuracy':f'{test["test"]["accuracy"]:.2%}' if test else 'Not evaluated'})
     st.dataframe(table,hide_index=True,width='stretch')
     chosen=st.selectbox('Detailed results',[r[0] for r in records],format_func=str.upper)
     name,best,hist,test=next(r for r in records if r[0]==chosen)
     subset=st.radio('Evaluation set',['Validation','Shenzhen test'],horizontal=True)
     metrics=best['validation'] if subset=='Validation' else test['test'] if test else None
     if metrics:
      cols=st.columns(4)
      for col,key,label in zip(cols,['precision','recall','f1','auc'],['TB precision','TB recall','F1 score','ROC AUC']):col.metric(label,f'{metrics[key]:.3f}')
      st.markdown('**Confusion matrix** — rows show the source label; columns show the prediction.')
      cm=metrics['confusion_matrix'];st.dataframe([{'Source label':'Normal','Predicted Normal':cm[0][0],'Predicted TB':cm[0][1]},{'Source label':'TB','Predicted Normal':cm[1][0],'Predicted TB':cm[1][1]}],hide_index=True,width='stretch')
      if test:st.caption(f'Validation–test accuracy gap: {test["accuracy_gap_percentage_points"]:.2f} percentage points.')
     else:st.info('This checkpoint has not been evaluated on Shenzhen yet.')
     st.subheader('Training history')
     st.line_chart([{'Epoch':r['epoch'],'Training loss':r['train_loss'],'Validation loss':r['validation']['loss']} for r in hist],x='Epoch',y=['Training loss','Validation loss'])
     st.caption('The minimum validation loss selects the checkpoint. Test results do not select the checkpoint epoch.')
     st.warning('Performance on Shenzhen can differ substantially from validation. Source differences, imbalanced classes and incomplete patient metadata limit generalization. Shenzhen was also evaluated in the earlier project; these are repeat experiments, not a newly unseen test.')
    else:
     st.markdown('<div class="eyebrow">PROJECT METHODOLOGY</div>',unsafe_allow_html=True)
     st.title('From image to prediction')
     st.markdown('### Data preparation\nTBX11K, cleaned Kaggle and Montgomery images form the development pool. An approximately 80/20 grouped split gives 6,508 training and 1,628 validation images. All 662 Shenzhen images are reserved for testing.')
     st.markdown('### Image processing\nGrayscale → resize to 224 × 224 → CLAHE → three-channel replication → divide by 255 → ImageNet standardization → CHW array.')
     st.markdown('### Feature fusion\nCNN features (256 values) + projected ViT CLS features (256 values) → combined vector (512 values) → classifier → two class scores.')
     st.markdown('### Training\nBatch size 16, gradient accumulation to 32, mixed precision and validation after each epoch. The pretrained ViT backbone is frozen; its projection and classifier learn from the training data. The hybrid CNN also learns.')
     st.markdown('### Interpretation\nClass 0 is Normal; class 1 follows the datasets’ TB labels. Those labels do not uniformly establish active disease. This is a research classifier, not a clinical diagnostic system.')
     for path,label in [(ROOT/'About-Project/about_dataset.docx','Dataset viva guide'),(ROOT/'About-Project/about_training.docx','Architecture and training guide')]:
      if path.exists():st.download_button(label,path.read_bytes(),path.name,'application/vnd.openxmlformats-officedocument.wordprocessingml.document')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--predict-worker':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        prediction_worker()
    else:
        main()
