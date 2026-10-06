"""Revalidate and export the user-authorized, explicitly unpromoted R25 control probe."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import config
from experiments import mask_correction_r17 as core
from experiments.evaluate_hbb_adaptation import load_model
from experiments.evaluate_proposal_adaptation import decode
from src.assemble_obb_gate_submission import assemble_image,decode_submission_rows
from src.dataset import read_gray_image
from src.frozen_mask_quality import Annotations,encode,score,terms
from src.predict_detect_refine import patch_torchvision_nms
from src.predict_detect_refine_multiscale import refine_boxes_tta,mask_nms_scored,make_masks_disjoint_scored

ROOT=Path('experiments/r25_probe_20260924')
OUTPUT=Path('submissions/submission_r25_control_r22_probe_20260924.csv')
PRIMARY=Path('experiments/aspect_refiner_r25_20260922/square_control.pth')
SHAPE=Path('models/crop_refiner_resnet34_512_consensus_ft.pth')
BASE=Path('submissions/submission_obb_gate_extras_pq04502_20260901.csv')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    resume=parser.parse_args().resume
    if OUTPUT.exists():raise FileExistsError('Output already exists')
    if ROOT.exists() and not resume:raise FileExistsError('Use --resume for existing validation')
    ROOT.mkdir(exist_ok=resume)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4);cv2.setNumThreads(2)
    core.ROOT=Path('experiments/mask_correction_r22_20260918')
    model,ckpt=load_model(PRIMARY);shape,_=load_model(SHAPE);corrector,_=core.load_corrector()
    baseline_hash=core.sha(BASE)
    opts=argparse.Namespace(box_pad=.5,batch_size=12,shape_weight=.2,threshold=.45,min_area=120,
        score_mode='detector',confirm_iou=.1,unmatched_min_area=3000,extra_match_iou=.1,extra_min_area=500)
    pre=dict(status='exploratory_not_promoted',primary_sha256=core.sha(PRIMARY),shape_sha256=core.sha(SHAPE),
             r22_sha256=core.sha(core.ROOT/'corrector.pth'),protected_baseline_sha256=baseline_hash,
             policy='Existing R25 square control; no hyperparameter changes; full batched validation before export.',
             export_sanity='Validation PQ must remain above saved R22; predicted TP must not decrease. This is not the production promotion gate.',
             uploaded=False,leaderboard_score=None)
    if resume:
        saved=json.loads((ROOT/'preregistration.json').read_text())
        for key in ['primary_sha256','shape_sha256','r22_sha256','protected_baseline_sha256']:
            assert saved[key]==pre[key], key
    else:
        (ROOT/'preregistration.json').write_text(json.dumps(pre,indent=2))
    def predict(image,boxes,scores,c,e):
        raw=refine_boxes_tta(model,image,np.asarray(boxes).reshape(-1,4),np.asarray(scores),'cuda',opts,512,shape) if len(boxes) else []
        hbb=[m for m,s in make_masks_disjoint_scored(mask_nms_scored(raw,.12),120)]
        assembled=assemble_image(hbb,c,e,opts)[0]
        return core.correct_image(corrector,image,assembled,1.)
    _,val=core.names();assert not set(val)&set(ckpt['train_file_names'])
    if resume:
        result=json.loads((ROOT/'validation.json').read_text())
        cached=pd.read_csv(ROOT/'counts.csv')
        assert cached.file_name.tolist()==val and len(cached)==141
        assert abs(score(cached[['iou_sum','tp','fp','fn']].to_numpy())['pq']-result['pq'])<1e-12
        assert result['delta_vs_r22']>0 and result['tp']>=443
    else:
        raw={r['file_name']:r for r in json.loads(Path('experiments/pipeline_diagnosis_r5/detector_boxes.json').read_text())}
        c=pd.read_csv('experiments/valpred_obb_refine_tta_c030.csv');e=pd.read_csv('experiments/valpred_obb_refine_tta_c038.csv')
        ann=Annotations(config.ANNOTATION_PATH);counts=[];valrows=[]
        for i,name in enumerate(val):
            image=read_gray_image(config.TRAIN_IMG_DIR/name);r=raw[name];keep=np.asarray(r['scores'])>=.22
            masks=predict(image,np.asarray(r['boxes']).reshape(-1,4)[keep],np.asarray(r['scores'])[keep],decode(c,name,image.shape),decode(e,name,image.shape))
            enc=[encode(m) for m in masks];counts.append(sum((terms(enc,g) for g in ann.get(name)),np.zeros(4)))
            valrows.extend(dict(file_name=name,filament_id=f'{Path(name).stem}_{j}',segmentation_rle=m['counts'].decode()) for j,m in enumerate(enc,1))
            if (i+1)%30==0:print('validation',i+1,len(val),flush=True)
        result=score(counts);ref=pd.read_csv(core.ROOT/'corrected_counts.csv') if (core.ROOT/'corrected_counts.csv').exists() else None
        if ref is None:raise FileNotFoundError('R22 per-image counts')
        reference=ref.set_index('file_name').loc[val,['iou_sum','tp','fp','fn']].to_numpy()
        rng=np.random.default_rng(20260924);deltas=[];array=np.asarray(counts)
        for _ in range(2000):
            idx=rng.integers(0,len(val),len(val));deltas.append(score(array[idx])['pq']-score(reference[idx])['pq'])
        result.update(delta_vs_r22=result['pq']-score(reference)['pq'],ci95=np.quantile(deltas,[.025,.975]).tolist(),
                      improved_blocks=sum(score(array[idx])['pq']>score(reference[idx])['pq'] for idx in np.array_split(np.arange(len(val)),4)))
        result['promote']=bool(result['delta_vs_r22']>=.002 and result['ci95'][0]>0 and result['improved_blocks']>=3)
        (ROOT/'validation.json').write_text(json.dumps(result,indent=2));pd.DataFrame(counts,columns=['iou_sum','tp','fp','fn']).assign(file_name=val).to_csv(ROOT/'counts.csv',index=False)
        pd.DataFrame(valrows).to_csv(ROOT/'valpred.csv',index=False)
        if result['delta_vs_r22']<=0 or result['tp']<score(reference)['tp']:raise RuntimeError('Probe failed export sanity check')
    patch_torchvision_nms()
    from ultralytics import YOLO
    detector=YOLO('models/yolo/yolo11s_boxes_1280_e25/weights/best.pt')
    c=pd.read_csv('submissions/submission_obb_refine_tta_c030.csv');e=pd.read_csv('submissions/submission_obb_refine_tta_c038.csv')
    rows=[];empty=[];files=sorted(p for p in config.TEST_IMG_DIR.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    cache=ROOT/'test_cache';cache.mkdir(exist_ok=True)
    for i,path in enumerate(files):
        cache_path=cache/(path.name+'.json')
        if cache_path.exists():
            item=json.loads(cache_path.read_text())
            assert item['file_name']==path.name
            rows.extend(item['rows'])
            if not item['rows']:empty.append(path.name)
            continue
        print('predict',i+1,path.name,flush=True)
        row_start=len(rows)
        image=read_gray_image(path);boxes=[];scores=[]
        for size in [1024,1280]:
            det=detector.predict(str(path),imgsz=size,conf=.22,iou=.45,device=0,verbose=False)[0]
            boxes.extend(det.boxes.xyxy.cpu().numpy());scores.extend(det.boxes.conf.cpu().numpy())
        masks=predict(image,boxes,scores,decode_submission_rows(c,path.stem,image.shape),decode_submission_rows(e,path.stem,image.shape))
        occupied=np.zeros(image.shape,np.uint8)
        for j,m in enumerate(masks,1):
            assert m.sum()>0 and not (occupied&m).any();occupied|=m
            rows.append(dict(filament_id=f'{path.stem}_{j}',segmentation_rle=encode(m)['counts'].decode()))
        cache_path.write_text(json.dumps(dict(file_name=path.name,rows=rows[row_start:])))
        if not masks:empty.append(path.name)
        if (i+1)%30==0:print('test',i+1,len(files),flush=True)
    frame=pd.DataFrame(rows,columns=['filament_id','segmentation_rle']);assert not frame.filament_id.duplicated().any() and not frame.isna().any().any()
    frame.to_csv(OUTPUT,index=False)
    assert core.sha(BASE)==baseline_hash and core.sha(PRIMARY)==pre['primary_sha256']
    destination=Path.home()/'Downloads'/OUTPUT.name
    if destination.exists():raise FileExistsError(destination)
    shutil.copyfile(OUTPUT,destination)
    audit=dict(**pre,validation=result,submission=str(OUTPUT),download_path=str(destination),sha256=core.sha(OUTPUT),
               test_images=len(files),nonempty_images=len(files)-len(empty),empty_images=empty,rows=len(frame),no_overlaps=True)
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
