"""Verify every saved stage against its preceding operation and a fresh raw-image pipeline.
This script does not train a model. It also creates a labeled visual walkthrough.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pipeline_utils import ROOT, MEAN, STD, read_csv, read_image, digest, write_json, stage_path

STAGES = ['CLEANED_DATASET', 'GRAYSCALE', 'RESIZING', 'CONTRAST_NORMALIZATION',
          'CHANNEL_REPLICATION', 'INTENSITY_SCALING', 'Z_SCORE_STANDARDIZATION', 'FINAL_DATASET']


def check_one(rows):
    original, gray_row, resize_row, clahe_row, rgb_row, scaled_row, z_row, final_row = rows
    raw = read_image(ROOT / original['original_path'], cv2.IMREAD_GRAYSCALE)
    sha = __import__('hashlib').sha256(str(raw.shape).encode() + raw.tobytes()).hexdigest()
    assert sha == original['pixel_sha256'], f'Original changed: {original["id"]}'
    gray = read_image(ROOT / gray_row['file_path'])
    resized = read_image(ROOT / resize_row['file_path'])
    contrast = read_image(ROOT / clahe_row['file_path'])
    rgb = read_image(ROOT / rgb_row['file_path'])
    scaled = np.load(ROOT / scaled_row['file_path'], allow_pickle=False)
    standardized = np.load(ROOT / z_row['file_path'], allow_pickle=False)
    final = np.load(ROOT / final_row['file_path'], allow_pickle=False)
    np.testing.assert_array_equal(gray, raw)
    np.testing.assert_array_equal(resized, cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA))
    np.testing.assert_array_equal(contrast, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(resized))
    np.testing.assert_array_equal(rgb, np.repeat(contrast[:, :, None], 3, axis=2))
    np.testing.assert_array_equal(scaled, rgb.astype(np.float32) / 255.0)
    np.testing.assert_array_equal(standardized, (scaled - MEAN) / STD)
    np.testing.assert_array_equal(final, standardized.transpose(2, 0, 1))
    assert final.shape == (3,224,224) and final.flags.c_contiguous
    for a in [scaled, standardized, final]:
        assert a.dtype == np.float32 and np.isfinite(a).all()
    assert scaled.min() >= 0 and scaled.max() <= 1
    # Independently recompute from RAW, without reading intermediate outputs.
    fresh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
        cv2.resize(raw, (224,224), interpolation=cv2.INTER_AREA))
    fresh = (np.repeat(fresh[:, :, None], 3, axis=2).astype(np.float32)/255.0 - MEAN)/STD
    np.testing.assert_array_equal(final, fresh.transpose(2,0,1))
    return float(final.min()), float(final.max())


def make_preview(manifests):
    # A training example avoids using test examples for presentation choices.
    row = next(r for r in manifests[0] if r['split']=='train' and r['label']=='1' and r['source']=='tbx11k')
    index = next(i for i,r in enumerate(manifests[0]) if r['id']==row['id'])
    arrays = []
    for stage in range(1,8):
        path=ROOT/manifests[stage][index]['file_path']
        arrays.append(np.load(path,allow_pickle=False) if path.suffix=='.npy' else read_image(path))
    gray,resized,contrast,rgb,scaled,z,final=arrays
    # Display inverses are ONLY for this figure; numerical stage files are unmodified.
    shown=[gray,resized,contrast,rgb,np.rint(scaled*255).astype(np.uint8),
           np.rint(np.clip(z*STD+MEAN,0,1)*255).astype(np.uint8),
           np.rint(np.clip(final.transpose(1,2,0)*STD+MEAN,0,1)*255).astype(np.uint8)]
    titles=['1. Grayscale','2. Resize','3. CLAHE','4. Three channels','5. Intensity scaling','6. Standardization','7. Final CHW']
    captions=['Original resolution','224 x 224; INTER_AREA','Clip 2.0; grid 8 x 8','224 x 224 x 3; uint8',
              'HWC float32; [0,1]','HWC float32; ImageNet','3 x 224 x 224; float32']
    canvas=Image.new('RGB',(1280,850),'#f4f7fb');draw=ImageDraw.Draw(canvas)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def make_font(size):
        return ImageFont.truetype(font,size) if Path(font).exists() else ImageFont.load_default()
    draw.text((24,18),'Preprocessing: the same X-ray through every saved stage',font=make_font(25),fill='#18324e')
    draw.text((24,54),row['id']+' | Training example | Labels describe stored data, not display thumbnails',font=make_font(16),fill='#536579')
    for i,(im,title,caption) in enumerate(zip(shown,titles,captions)):
        x=24+(i%4)*314;y=100+(i//4)*345
        draw.text((x,y),title,font=make_font(21),fill='#18324e')
        pil=Image.fromarray(im).convert('RGB');pil.thumbnail((280,260))
        canvas.paste(pil,(x+(280-pil.width)//2,y+35+(260-pil.height)//2))
        draw.text((x,y+302),caption,font=make_font(15),fill='#536579')
    draw.text((966,445),'Follow one pixel',font=make_font(21),fill='#18324e')
    notes=['Position: row112, column112', f'Resized: {int(resized[112,112])}',
           f'After CLAHE: {int(contrast[112,112])}', f'Scaled: {float(scaled[112,112,0]):.6f}',
           'Standardized channels:', ', '.join(f'{v:.3f}' for v in z[112,112]), 'Final: same values, CHW layout']
    for i,note in enumerate(notes):
        draw.text((966,490+i*30),note,font=make_font(16),fill='#536579')
    draw.text((24,805),'Float-array panels are decoded for viewing; scaling/standardization do not add another contrast enhancement.',font=make_font(16),fill='#536579')
    canvas.save(ROOT/'Preprocessing'/'PREPROCESSING_PREVIEW.png')
    write_json(ROOT/'PIXEL_WALKTHROUGH.json',dict(id=row['id'],position_yx=[112,112],
        resized_pixel=int(resized[112,112]),clahe_pixel=int(contrast[112,112]),
        replicated_pixel=rgb[112,112].tolist(),scaled_pixel=scaled[112,112].tolist(),
        standardized_pixel=z[112,112].tolist(),final_pixel=final[:,112,112].tolist(),
        note='Coordinates refer to resized224x224 images. Preview reverses scaling/standardization for display.'))


def main():
    cv2.setNumThreads(1)
    manifests=[]
    for i,stage in enumerate(STAGES):
        folder=stage_path(stage)
        info=json.loads((folder/'stage_info.json').read_text())
        rows=read_csv(folder/'manifest.csv')
        assert info['status']=='complete' and info['count']==len(rows)
        assert info['manifest_sha256']==digest(folder/'manifest.csv')
        paths=[ROOT/r['file_path'] for r in rows]
        assert len({r['id'] for r in rows})==len(rows) and all(p.is_file() for p in paths)
        extension='.png' if i<5 else '.npy'
        assert len(list(folder.glob(f'*/*/*{extension}')))==len(rows)
        if manifests:
            assert info['source_manifest_sha256']==digest(stage_path(STAGES[i-1])/'manifest.csv')
            for old,new in zip(manifests[0],rows):
                assert {k:v for k,v in old.items() if k!='file_path'}=={k:v for k,v in new.items() if k!='file_path'}
        manifests.append(rows)
    rows=manifests[0]
    for key in ['group','pixel_sha256']:
        sets=[{r[key] for r in rows if r['split']==split} for split in ['train','val','test']]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert all((r['source']=='shenzhen') == (r['split']=='test') for r in rows)
    checked=0;minimum=float('inf');maximum=float('-inf')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for low,high in pool.map(check_one,zip(*manifests)):
            minimum=min(minimum,low);maximum=max(maximum,high);checked+=1
            if checked%500==0 or checked==len(rows): print(f'Verified all stages for {checked}/{len(rows)} images',flush=True)
    make_preview(manifests)
    report=dict(status='passed',verified_images=checked,verified_stages=STAGES,
        counts=dict(Counter(f'{r["split"]}:{r["label"]}' for r in rows)),
        final_shape=[3,224,224],dtype='float32',final_min=minimum,final_max=maximum,
        exact_stage_math=True,final_equals_fresh_raw_pipeline=True,original_pixel_hashes_unchanged=True,
        group_and_hash_splits_disjoint=True,training_performed=False,finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/'VERIFICATION_REPORT.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
