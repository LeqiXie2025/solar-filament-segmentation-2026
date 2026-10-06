"""Paired full-image test of frozen versus adapted refiner, without threshold tuning."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as coco
import torch

import config
from src.assemble_obb_gate_submission import assemble_image, greedy_pairs
from src.dataset import read_gray_image
from src.frozen_mask_quality import Annotations, encode, score, terms
from src.model_factory import create_model_from_checkpoint
from src.predict_detect_refine import patch_torchvision_nms, time_val_names
from src.predict_detect_refine_multiscale import mask_nms_scored, make_masks_disjoint_scored
from src.predict_obb_refine import refine_obbs


def decode(frame, name, shape):
    return [coco.decode(dict(size=shape, counts=v.encode())).astype(np.uint8)
            for v in frame.loc[frame.file_name==name,'segmentation_rle']]


def replace_boundaries(base, proposals):
    # Only exchange one-to-one strongly overlapping instances; no label access.
    pairs, _, _ = greedy_pairs(base, proposals, .5)
    replacements = {i:proposals[j] for i,j,_ in pairs}
    candidates = [(replacements.get(i, mask), 1.) for i,mask in enumerate(base)]
    return [m for m,_ in make_masks_disjoint_scored(candidates,80)]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapted',type=Path,default=Path('models/proposal_adapted_refiner_r3.pth'))
    p.add_argument('--output',type=Path,default=Path('experiments/proposal_adapted_r3'))
    a=p.parse_args()
    if (a.output/'comparison.csv').exists():
        raise FileExistsError(a.output/'comparison.csv')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device=torch.device('cuda')
    patch_torchvision_nms()
    from ultralytics import YOLO
    detector=YOLO('models/yolo/yolo11s_obb_medoid_1280_e20/weights/best.pt')
    checkpoints={key:torch.load(path,map_location='cpu',weights_only=False) for key,path in
                 [('original','models/obb_refiner_resnet34_768x256_iou_boundary_skeleton_e8.pth'),('adapted',a.adapted)]}
    models={}
    for key,checkpoint in checkpoints.items():
        model=create_model_from_checkpoint(checkpoint).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        models[key]=model
    names=sorted(time_val_names(config.ANNOTATION_PATH,{p.name for p in config.TRAIN_IMG_DIR.iterdir()}))
    assert not (set(checkpoints['adapted']['train_file_names'])&set(names))
    annotation=Annotations(config.ANNOTATION_PATH)
    hbb=pd.read_csv('experiments/valpred_yolo11s_c022.csv')
    saved=pd.read_csv('experiments/obb_continuation_20260906/selected_valpred.csv')
    options=argparse.Namespace(pad_long=None,pad_short=None,threshold=.45,min_area=120,batch_size=12,
                               refiner_tta=True,confirm_iou=.10,unmatched_min_area=3000,extra_match_iou=.10,extra_min_area=500)
    policies=['saved_baseline','original_assembled','adapted_assembled','adapted_boundaries', 'original_obb','adapted_obb']
    totals={key:[] for key in policies}
    predictions={key:[] for key in policies}
    for step,name in enumerate(names):
        path=config.TRAIN_IMG_DIR/name
        image=read_gray_image(path)
        detection=detector.predict(str(path),conf=.30,iou=.45,imgsz=1280,device=0,verbose=False)[0]
        quads=detection.obb.xyxyxyxy.cpu().numpy() if detection.obb is not None else []
        scores=detection.obb.conf.cpu().numpy() if detection.obb is not None else []
        hb=decode(hbb,name,image.shape)
        outputs={'saved_baseline':decode(saved,name,image.shape)}
        for key,model in models.items():
            raw=refine_obbs(model,image,quads,scores,device,options,checkpoints[key]) if len(quads) else []
            masks={}
            for threshold in [.30,.38]:
                filtered=mask_nms_scored([(m,s) for m,s in raw if s>=threshold],.12)
                masks[threshold]=[m for m,_ in make_masks_disjoint_scored(filtered,120)]
            outputs[key+'_obb']=masks[.30]
            outputs[key+'_assembled']=assemble_image(hb,masks[.30],masks[.38],options)[0]
            if key=='adapted':
                outputs['adapted_boundaries']=replace_boundaries(outputs['saved_baseline'],masks[.30])
        groups=annotation.get(name)
        for key in policies:
            encoded=[encode(mask) for mask in outputs[key]]
            total=sum((terms(encoded,g) for g in groups),np.zeros(4))
            totals[key].append(total)
            for i,mask in enumerate(encoded,1):
                predictions[key].append(dict(file_name=name,filament_id=f'{path.stem}_{i}',segmentation_rle=mask['counts'].decode()))
        if (step+1)%20==0:
            print(f'images={step+1}/{len(names)} '+json.dumps({k:score(v)['pq'] for k,v in totals.items()}),flush=True)
    table=[]
    blocks=np.array_split(np.arange(len(names)),4)
    for key in policies:
        values=np.asarray(totals[key]); sums=values.sum(axis=0)
        table.append(dict(policy=key,**score(values),sq=sums[0]/max(1,sums[1]),
                          dq=sums[1]/max(1,sums[1]+.5*(sums[2]+sums[3])),
                          **{f'block_{i+1}':score(values[b])['pq'] for i,b in enumerate(blocks)}))
        pd.DataFrame(values,columns=['iou_sum','tp','fp','fn']).assign(file_name=names).to_csv(a.output/f'{key}_counts.csv',index=False)
        pd.DataFrame(predictions[key],columns=['file_name','filament_id','segmentation_rle']).to_csv(a.output/f'{key}_valpred.csv',index=False)
    table=pd.DataFrame(table)
    table.to_csv(a.output/'comparison.csv',index=False)
    rng=np.random.default_rng(20260908)
    differences={key:[] for key in ['adapted_assembled','adapted_boundaries']}
    for _ in range(2000):
        idx=rng.integers(0,len(names),len(names))
        base=score(np.asarray(totals['saved_baseline'])[idx])['pq']
        for key in differences:
            differences[key].append(score(np.asarray(totals[key])[idx])['pq']-base)
    summary=dict(images=len(names),primary_policy='adapted_assembled',diagnostic_policy='adapted_boundaries',
                 paired_delta_95={k:np.quantile(v,[.025,.975]).tolist() for k,v in differences.items()},
                 warning='Fixed threshold policies. Historically reused validation; bootstrap not selection-adjusted. No test labels or leaderboard tuning.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(table.to_string(index=False),flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
