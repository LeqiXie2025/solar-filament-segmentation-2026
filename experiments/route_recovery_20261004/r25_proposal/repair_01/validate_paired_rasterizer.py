"""Frozen full-route validation of paired R25 target-rasterizer arms; COCO raw-record metric."""
import json, hashlib, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from collections import defaultdict
import cv2, numpy as np, pandas as pd, torch, config
from pycocotools import mask as coco
from experiments import mask_correction_r17 as core
from experiments.evaluate_hbb_adaptation import load_model
from experiments.evaluate_proposal_adaptation import decode
from experiments.aspect_refiner_r25 import inputs, restore, probabilities, paste
from src.crop_refiner_utils import crop_with_padding
from src.dataset import read_gray_image
from src.frozen_mask_quality import encode, score, terms
from src.assemble_obb_gate_submission import assemble_image
from src.predict_detect_refine import patch_torchvision_nms
from src.predict_detect_refine_multiscale import refine_boxes_tta, mask_nms_scored, make_masks_disjoint_scored
from src.gt_dedup import _decode_segmentation

ROOT=Path('.')
OUT=Path('experiments/route_recovery_20261004/r25_proposal/repair_01')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode_gt_coco(seg,h,w):
    if isinstance(seg,list):
        if not seg:return np.zeros((h,w),np.uint8)
        rs=coco.frPyObjects(seg,h,w); rr=coco.merge(rs) if rs else None
        return coco.decode(rr).astype(np.uint8) if rr is not None else np.zeros((h,w),np.uint8)
    if isinstance(seg,dict):
        r=dict(seg)
        if isinstance(r.get('counts'),str):r['counts']=r['counts'].encode()
        return coco.decode(r).astype(np.uint8)
    return np.zeros((h,w),np.uint8)
class CocoRecords:
    def __init__(self,path):
        raw=json.loads(Path(path).read_text(encoding='utf-8')); imgs=defaultdict(list); anns=defaultdict(list)
        for im in raw['images']:imgs[im['file_name']].append(im)
        for a in raw['annotations']:anns[a['image_id']].append(a)
        self.groups={}
        for name,records in imgs.items():
            self.groups[name]=[]
            for im in records:
                masks=[]
                for a in anns[im['id']]:
                    m=decode_gt_coco(a['segmentation'],im['height'],im['width'])
                    if m.any():masks.append(encode(m))
                self.groups[name].append(masks)

def counts_for(masks,groups): return sum((terms(masks,g) for g in groups),np.zeros(4))
def read_masks(frame,name):
    return [coco.decode({'size':[2048,2048],'counts':v.encode()}).astype(np.uint8) for v in frame.loc[frame.file_name==name,'segmentation_rle']]
def cluster(name): return str(name)[:8]
def bootstrap(a,b,names,reps=3000,seed=40404):
    dates=sorted({cluster(x) for x in names}); bydate={d:[i for i,n in enumerate(names) if cluster(n)==d] for d in dates}; rng=np.random.default_rng(seed); dif=[]
    for _ in range(reps):
        picked=rng.choice(dates,len(dates),replace=True); ix=[i for d in picked for i in bydate[d]]
        dif.append(score(a[ix])['pq']-score(b[ix])['pq'])
    return np.quantile(dif,[.025,.975]).tolist()

@torch.no_grad()
def main():
    pre=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
    for k,rel in pre['sources']['path_by_key'].items():assert sha(rel)==pre['sources']['sha256_by_key'][k],rel
    arm_ckpts={k:OUT/f'{k}_epoch6.pth' for k in ['control_cv2','candidate_coco']}
    for p in arm_ckpts.values():assert p.exists()
    core.ROOT=Path('experiments/mask_correction_r22_20260918')
    corrector,_=core.load_corrector()
    shape,_=load_model('models/crop_refiner_resnet34_512_consensus_ft.pth')
    models={k:load_model(v)[0] for k,v in arm_ckpts.items()}
    raw={r['file_name']:r for r in json.loads(Path('experiments/pipeline_diagnosis_r5/detector_boxes.json').read_text(encoding='utf-8'))}
    c=pd.read_csv('experiments/valpred_obb_refine_tta_c030.csv');e=pd.read_csv('experiments/valpred_obb_refine_tta_c038.csv')
    opts=type('Opts',(),dict(box_pad=.5,batch_size=12,shape_weight=.2,threshold=.45,min_area=120,score_mode='detector',confirm_iou=.1,unmatched_min_area=3000,extra_match_iou=.1,extra_min_area=500))()
    confirm_names={k for k in raw}; annotations=CocoRecords(config.ANNOTATION_PATH)
    val=sorted(set(pre['split']['validation_files'])); assert len(val)==141 and set(val)<=confirm_names
    patch_torchvision_nms(); counts={k:[] for k in models}; rows={k:[] for k in models}; start=time.time()
    cache_path=OUT/'paired_val_image_cache.jsonl'; cache={}
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line); cache[item['file_name']]=item
        assert set(cache)<=set(val)
    for ii,name in enumerate(val):
        if name in cache:
            item=cache[name]
            for arm in models:
                counts[arm].append(np.asarray(item['counts'][arm],float)); rows[arm].extend(item['rows'][arm])
            continue
        image=read_gray_image(config.TRAIN_IMG_DIR/name); rr=raw[name]; keep=np.asarray(rr['scores'])>=.22; boxes=np.asarray(rr['boxes']).reshape(-1,4)[keep]; scores=np.asarray(rr['scores'])[keep]
        cand={k:[] for k in models}
        if len(boxes):
            raw_by_arm={k:refine_boxes_tta(model,image,boxes,scores,'cuda',opts,512,shape) for k,model in models.items()}
        else:raw_by_arm={k:[] for k in models}
        item={'file_name':name,'counts':{},'rows':{}}
        for arm in models:
            hbb=[m for m,s in make_masks_disjoint_scored(mask_nms_scored(raw_by_arm[arm],.12),120)]
            assembled=assemble_image(hbb,decode(c,name,image.shape),decode(e,name,image.shape),opts)[0]
            fixed=core.correct_image(corrector,image,assembled,1.)
            enc=[encode(m) for m in fixed]; item['counts'][arm]=counts_for(enc,annotations.groups[name]).tolist()
            item['rows'][arm]=[{'file_name':name,'filament_id':f'{Path(name).stem}_{j}','segmentation_rle':m['counts'].decode()} for j,m in enumerate(enc,1)]
            counts[arm].append(np.asarray(item['counts'][arm],float)); rows[arm].extend(item['rows'][arm])
        with cache_path.open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
        cache[name]=item
        if (ii+1)%20==0:print(json.dumps({'phase':'validation','images':ii+1,'total':len(val),'seconds':round(time.time()-start)}),flush=True)
    arrays={k:np.asarray(v,float) for k,v in counts.items()}; baseline=pd.read_csv('experiments/r25_probe_20260924/valpred.csv'); bcounts=[]
    for name in val:bcounts.append(counts_for([encode(m) for m in read_masks(baseline,name)],annotations.groups[name]))
    barray=np.asarray(bcounts,float)
    # Ensure frozen original-route COCO rescore exactly reproduces prereg baseline before comparing arms.
    base_score=score(barray); expected=pre['validation']['frozen_original_fullroute_coco_baseline']
    assert base_score['tp']==expected['tp'] and base_score['fp']==expected['fp'] and base_score['fn']==expected['fn'] and abs(base_score['pq']-expected['pq'])<1e-8,(base_score,expected)
    for k in models:
        pd.DataFrame(arrays[k],columns=['iou_sum','tp','fp','fn']).assign(file_name=val).to_csv(OUT/f'{k}_counts.csv',index=False)
        pd.DataFrame(rows[k]).to_csv(OUT/f'{k}_valpred.csv',index=False)
    base=score(barray);ctrl=score(arrays['control_cv2']);cand=score(arrays['candidate_coco']);ci=bootstrap(arrays['candidate_coco'],arrays['control_cv2'],val)
    delta=cand['pq']-ctrl['pq']; fpmax=base['fp']*1.05
    gate={'candidate_control_delta_min':delta>=.002,'candidate_control_ci_lower_positive':ci[0]>0,'candidate_tp_at_least_frozen_original':cand['tp']>=base['tp'],'candidate_fp_within_5pct_original':cand['fp']<=fpmax,'candidate_pq_at_least_original':cand['pq']>=base['pq']}
    result={'status':'complete','metric':'COCO raw annotation records, all-hit IoU>0.5 pooled PQ','images':len(val),'dates':len({cluster(n) for n in val}),'baseline':base,'control_cv2':ctrl,'candidate_coco':cand,'candidate_minus_control_pq':delta,'date_cluster_ci95':ci,'gates':gate,'promote_to_test':all(gate.values()),'validation_exposure':'R25 validation is exposed development, not OOF.','source_hashes':{str(p):sha(p) for p in arm_ckpts.values()},'valpred_paths':{k:str(OUT/f'{k}_valpred.csv') for k in models}}
    (OUT/'validation_gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()





