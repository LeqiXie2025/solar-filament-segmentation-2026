"""Paired HBB-refiner substitution through the existing final assembly."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import config
from experiments.evaluate_proposal_adaptation import decode
from src.assemble_obb_gate_submission import assemble_image
from src.dataset import read_gray_image
from src.frozen_mask_quality import Annotations, encode, score, terms
from src.model_factory import create_model_from_checkpoint
from src.proposal_classifier import create_proposal_classifier
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox
from src.predict_detect_refine import patch_torchvision_nms,time_val_names
from src.predict_detect_refine_multiscale import refine_boxes_tta,mask_nms_scored,make_masks_disjoint_scored


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=create_model_from_checkpoint(checkpoint).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    return model,checkpoint


def load_classifier(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=create_proposal_classifier(pretrained=False).cuda().eval()
    model.load_state_dict(ckpt['model'])
    return model,ckpt


@torch.no_grad()
def classifier_scores(model, image, boxes, image_size, pad):
    if len(boxes)==0: return np.empty(0,dtype=np.float32)
    chunks=[]
    for box in boxes:
        x,y,x1,y1=map(float,box)
        bbox=(x,y,x1-x,y1-y)
        bounds=crop_bounds_from_bbox(bbox,image.shape[0],image.shape[1],pad=pad,jitter=False)
        crop=cv2.resize(crop_with_padding(image,bounds),(image_size,image_size),interpolation=cv2.INTER_LINEAR)
        seed=seed_from_bbox(bbox,bounds,image_size)
        chunks.append(torch.from_numpy(np.ascontiguousarray(make_refiner_channels(crop,seed))).float())
    return torch.sigmoid(model(torch.stack(chunks).cuda())[:,0]).cpu().numpy()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shape',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('experiments/hbb_adapted_r3'))
    p.add_argument('--adapted',type=Path,default=Path('models/hbb_proposal_adapted_r3.pth'))
    p.add_argument('--control',type=Path)
    p.add_argument('--classifier',type=Path)
    p.add_argument('--classifier-threshold',type=float,default=.35)
    a=p.parse_args()
    if (a.output/'comparison.csv').exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    patch_torchvision_nms()
    from ultralytics import YOLO
    detector=YOLO('models/yolo/yolo11s_boxes_1280_e25/weights/best.pt')
    models={k:load_model(v) for k,v in [('original','models/crop_refiner_resnet34_512_ioubce_ft.pth'),
                                      ('adapted',a.adapted)]}
    if a.control:
        models['control']=load_model(a.control)
    classifier=load_classifier(a.classifier) if a.classifier else None
    shape,_=load_model(a.shape)
    options=argparse.Namespace(box_pad=.5,batch_size=12,shape_weight=.2,threshold=.45,min_area=120,score_mode='detector',
                               confirm_iou=.1,unmatched_min_area=3000,extra_match_iou=.1,extra_min_area=500)
    names=sorted(time_val_names(config.ANNOTATION_PATH,{p.name for p in config.TRAIN_IMG_DIR.iterdir()}))
    for key in models:
        if key != 'original':
            assert not set(names)&set(models[key][1]['train_file_names'])
    cached=pd.read_csv('experiments/obb_continuation_20260906/selected_valpred.csv')
    confirm=pd.read_csv('experiments/valpred_obb_refine_tta_c030.csv')
    extras=pd.read_csv('experiments/valpred_obb_refine_tta_c038.csv')
    annotation=Annotations(config.ANNOTATION_PATH)
    policies=['cached',*models,*[k+'_hbb' for k in models]]
    counts={key:[] for key in policies}; rows={key:[] for key in policies}
    for i,name in enumerate(names):
        image=read_gray_image(config.TRAIN_IMG_DIR/name)
        boxes,scores=[],[]
        for size in [1024,1280]:
            det=detector.predict(str(config.TRAIN_IMG_DIR/name),imgsz=size,conf=.22,iou=.45,device=0,verbose=False)[0]
            boxes.extend(det.boxes.xyxy.cpu().numpy()); scores.extend(det.boxes.conf.cpu().numpy())
        if classifier:
            keep=classifier_scores(classifier[0],image,np.asarray(boxes),classifier[1].get('image_size',128),classifier[1].get('pad',.5)) >= a.classifier_threshold
            boxes=np.asarray(boxes)[keep]; scores=np.asarray(scores)[keep]
        c,e=decode(confirm,name,image.shape),decode(extras,name,image.shape)
        outputs={'cached':decode(cached,name,image.shape)}
        for key,(model,ckpt) in models.items():
            raw=refine_boxes_tta(model,image,np.asarray(boxes),np.asarray(scores),'cuda',options,ckpt['image_size'],shape) if len(boxes) else []
            hb=[m for m,s in make_masks_disjoint_scored(mask_nms_scored(raw,.12),120)]
            outputs[key+'_hbb']=hb
            outputs[key]=assemble_image(hb,c,e,options)[0]
        groups=annotation.get(name)
        for key,masks in outputs.items():
            encoded=[encode(m) for m in masks]
            counts[key].append(sum((terms(encoded,g) for g in groups),np.zeros(4)))
            for j,m in enumerate(encoded,1):
                rows[key].append(dict(file_name=name,filament_id=f'{Path(name).stem}_{j}',segmentation_rle=m['counts'].decode()))
        if (i+1)%20==0:
            print(f'images={i+1}/{len(names)} '+json.dumps({k:score(v)['pq'] for k,v in counts.items()}),flush=True)
    table=[]; blocks=np.array_split(np.arange(len(names)),4)
    for key in policies:
        values=np.asarray(counts[key]); summed=values.sum(0)
        table.append(dict(policy=key,**score(values),sq=summed[0]/max(1,summed[1]),
                          **{f'block_{i+1}':score(values[b])['pq'] for i,b in enumerate(blocks)}))
        pd.DataFrame(values,columns=['iou_sum','tp','fp','fn']).assign(file_name=names).to_csv(a.output/f'{key}_counts.csv',index=False)
        pd.DataFrame(rows[key],columns=['file_name','filament_id','segmentation_rle']).to_csv(a.output/f'{key}_valpred.csv',index=False)
    rng=np.random.default_rng(20260908); delta=[]
    controls=[]; versus_control=[]
    for _ in range(2000):
        idx=rng.integers(0,len(names),len(names))
        delta.append(score(np.asarray(counts['adapted'])[idx])['pq']-score(np.asarray(counts['original'])[idx])['pq'])
        if a.control:
            controls.append(score(np.asarray(counts['control'])[idx])['pq']-score(np.asarray(counts['original'])[idx])['pq'])
            versus_control.append(score(np.asarray(counts['adapted'])[idx])['pq']-score(np.asarray(counts['control'])[idx])['pq'])
    result=dict(images=len(names),shape=str(a.shape),adapted=str(a.adapted),paired_delta_95=np.quantile(delta,[.025,.975]).tolist(),
                warning='Repeated historical validation, fixed epoch 2 and inference settings; no leaderboard tuning.')
    if a.control:
        result.update(control=str(a.control),control_delta_95=np.quantile(controls,[.025,.975]).tolist(),
                      adapted_vs_control_delta_95=np.quantile(versus_control,[.025,.975]).tolist())
    pd.DataFrame(table).to_csv(a.output/'comparison.csv',index=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(pd.DataFrame(table).to_string(index=False),flush=True); print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
