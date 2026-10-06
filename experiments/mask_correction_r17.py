"""Isolated residual mask correction; no upstream model or submission replacement."""
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as coco
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import config
from src.crop_refiner_utils import crop_bounds_from_bbox, crop_with_padding, make_refiner_channels, seed_from_bbox
from src.dataset import read_gray_image
from src.frozen_mask_quality import Annotations, encode, matrix, score, terms
from src.predict_detect_refine import time_val_names

ROOT = Path('experiments/mask_correction_r17_20260916')
SOURCE = Path('experiments/frozen_rescue_r2/train.jsonl')
BASE = Path('experiments/hbb_hard_positive_r16_20260915/original_valpred.csv')
SEED = 20260916
SIZE = 256
CROP_PAD = .15
MAX_CANDIDATES = 8
BATCH_SIZE = 16
BASE_LOGIT = 1.
RESIDUAL_LIMIT = 3.
RESIDUAL_PENALTY = .015
EXPORT_PATH = Path('submissions/submission_mask_correction_r17_20260916.csv')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def names():
    all_names = {p.name for p in config.TRAIN_IMG_DIR.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    val = set(time_val_names(config.ANNOTATION_PATH, all_names))
    return sorted(all_names-val), sorted(val)


def predictions(frame, name, shape):
    return [dict(size=list(shape), counts=str(s).encode()) for s in frame.loc[frame.file_name==name,'segmentation_rle']]


def features(image, initial, target=None):
    box = coco.toBbox(encode(initial)).tolist()
    bounds = crop_bounds_from_bbox(box,*image.shape,pad=CROP_PAD,jitter=False)
    gray = cv2.resize(crop_with_padding(image,bounds),(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
    prior = cv2.resize(crop_with_padding(initial,bounds),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
    prompt = seed_from_bbox(box,bounds,SIZE)
    channels = make_refiner_channels(gray,prompt)
    x = np.stack([channels[0],channels[1],prior,prompt])
    y = None if target is None else cv2.resize(crop_with_padding(target,bounds),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
    return np.round(np.clip(x,0,1)*255).astype(np.uint8), y, bounds


class Block(nn.Sequential):
    def __init__(self, ins, outs):
        super().__init__(nn.Conv2d(ins,outs,3,padding=1),nn.GroupNorm(4,outs),nn.SiLU(),
                         nn.Conv2d(outs,outs,3,padding=1),nn.GroupNorm(4,outs),nn.SiLU())


class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=Block(4,16); self.enc2=Block(16,32); self.enc3=Block(32,64)
        self.mid=Block(64,96); self.dec3=Block(160,64); self.dec2=Block(96,32); self.dec1=Block(48,16)
        self.out=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self,x):
        a=self.enc1(x); b=self.enc2(F.avg_pool2d(a,2)); c=self.enc3(F.avg_pool2d(b,2))
        z=self.mid(F.avg_pool2d(c,2))
        z=self.dec3(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        z=self.dec2(torch.cat([F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        z=self.dec1(torch.cat([F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return RESIDUAL_LIMIT*torch.tanh(self.out(z))


def corrected_mask(initial, delta, bounds, strength):
    if not np.isfinite(delta).all() or not np.isfinite(strength):
        raise ValueError('Nonfinite correction')
    if strength == 0:
        return initial.copy()
    x0,y0,x1,y1=bounds
    full=cv2.resize(delta,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
    left,top=max(0,x0),max(0,y0); right,bottom=min(initial.shape[1],x1),min(initial.shape[0],y1)
    out=initial.copy()
    local=BASE_LOGIT*(initial[top:bottom,left:right].astype(np.float32)*2-1)
    out[top:bottom,left:right]=(local+strength*full[top-y0:bottom-y0,left-x0:right-x0]>0)
    if not .5*initial.sum() <= out.sum() <= 2*initial.sum():
        return initial.copy()
    return out.astype(np.uint8)


def disjoint_corrections(original, proposed):
    if len(original)!=len(proposed):
        raise ValueError('Instance count changed')
    if not original:
        return []
    occupied=np.zeros_like(original[0],bool)
    for mask in original:
        if (occupied & mask.astype(bool)).any():
            raise ValueError('Baseline overlaps')
        occupied |= mask.astype(bool)
    claimed=np.zeros_like(occupied)
    result=[]
    for old,new in zip(original,proposed):
        safe=new.astype(bool) & ~(occupied & ~old.astype(bool)) & ~claimed
        if not .5*old.sum() <= safe.sum() <= 2*old.sum():
            safe=old.astype(bool).copy()
        assert safe.any() and not (safe & claimed).any()
        result.append(safe.astype(np.uint8)); claimed |= safe
    return result


def audit():
    train,val=names()
    records=[]
    for p in [Path('models/crop_refiner_resnet34_512_ioubce_ft.pth'),
              Path('models/crop_refiner_resnet34_512_consensus_ft.pth'),
              Path('models/hbb_hard_positive_r16_20260915.pth'),
              Path('models/hbb_hard_positive_control_r16_20260915.pth')]:
        c=torch.load(p,map_location='cpu',weights_only=False)
        history=p.with_suffix('.history.json')
        h=json.loads(history.read_text()) if history.exists() else {}
        records.append(dict(path=str(p),sha256=sha(p),arch=c.get('arch'),
            checkpoint_fixed_final_epoch=c.get('fixed_final_epoch'),
            history_last_epoch=h.get('epochs',[{}])[-1].get('epoch'),
            known_train_files=len(c.get('train_file_names',[])),
            known_final_val_overlap=len(set(val)&set(c.get('train_file_names',[]))),
            lineage_verified=bool(c.get('train_file_names'))))
    submissions=[dict(path=str(p),sha256=sha(p),rows=len(pd.read_csv(p))) for p in [
        Path('submissions/submission_obb_gate_extras_pq04502_20260901.csv'),
        Path('submissions/submission_frozen_rescue_pq045138_20260908.csv')]]
    supplied=[]
    for filename in ['submission_steps3_6_quality_gate (1).csv','submission_frozen_rescue_pq045138_20260908.csv',
                     'submission_current_best_obb_gate_pq04502.csv']:
        p=Path.home()/'Downloads'/filename
        if p.exists():
            digest=sha(p)
            supplied.append(dict(path=str(p),sha256=digest,user_reported_lb=.37,
                matches=[r['path'] for r in submissions if r['sha256']==digest]))
    save('audit.json',dict(checkpoints=records,submissions=submissions,supplied_submissions=supplied,train_images=len(train),val_images=len(val),
         r16_warning_error='Evaluator hard-coded epoch 2; both saved R16 checkpoints are actually fixed epoch 3.',
         leaderboard_file_identity='User supplied 0.37 CSVs hash-match local artifacts. Screenshot shows named OBB gate submission at 0.37; score not freshly fetched from Kaggle.',
         baseline_sha256=sha(BASE),source_sha256=sha(SOURCE),
         original_refiner_lineage='Checkpoint lacks train/val names; historical validation was also used for selection.'))
    annotation=Annotations(config.ANNOTATION_PATH); frame=pd.read_csv(BASE)
    rows=[]; counts={k:[] for k in ['baseline','oracle_remove_unmatched','oracle_repair_overlap','oracle_add_missed']}
    examples=[]
    for name in val:
        info=annotation.images[name][0]; shape=(info['height'],info['width'])
        preds=predictions(frame,name,shape)
        for group,gt in enumerate(annotation.get(name)):
            overlap=matrix(preds,gt); base=terms(preds,gt); counts['baseline'].append(base)
            matched=overlap>.5
            best=overlap.max(1) if gt else np.zeros(len(preds))
            counts['oracle_remove_unmatched'].append(terms([p for i,p in enumerate(preds) if best[i]>.5],gt))
            fixed=[dict(p) for p in preds]; used=set(np.where(matched.any(0))[0])
            candidates=sorted([(float(overlap[i,j]),i,j) for i in range(len(preds)) for j in range(len(gt))
                               if best[i]<=.5 and j not in used and .1<overlap[i,j]<=.5],reverse=True)
            repaired=set()
            for ov,i,j in candidates:
                if i not in repaired and j not in used:
                    fixed[i]=gt[j]; repaired.add(i); used.add(j)
            counts['oracle_repair_overlap'].append(terms(fixed,gt))
            missed=[gt[j] for j in range(len(gt)) if not matched[:,j].any()]
            counts['oracle_add_missed'].append(terms(preds+missed,gt))
            for i,b in enumerate(best):
                category='matched' if b>.5 else ('zero_overlap' if b==0 else 'low_overlap' if b<=.1 else 'overlap_below_match')
                rows.append(dict(file_name=name,annotator_group=group,instance=i,max_iou=float(b),category=category))
                if category=='overlap_below_match' and len(examples)<12 and group==0:
                    examples.append((name,preds[i],gt[int(overlap[i].argmax())],float(b)))
    pd.DataFrame(rows).to_csv(ROOT/'error_instances.csv',index=False)
    save('diagnosis.json',dict(categories=dict(Counter(r['category'] for r in rows)),
         oracle_scenarios={k:score(v) for k,v in counts.items()},
         stage_diagnosis=json.loads(Path('experiments/pipeline_diagnosis_r5/summary.json').read_text()),
         warning='Oracles use per-annotator validation labels and can overlap or be mutually inconsistent. Diagnostic only; not attainable model scores.'))
    tiles=[]
    for name,p,g,iou in examples:
        image=read_gray_image(config.TRAIN_IMG_DIR/name); pm=coco.decode(p); gm=coco.decode(g)
        box=coco.toBbox(encode(pm|gm)); bounds=crop_bounds_from_bbox(box,*image.shape,pad=.2)
        gray=cv2.resize(crop_with_padding(image,bounds),(280,250))
        tile=cv2.cvtColor((gray*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        for mask,color in [(pm,(0,0,255)),(gm,(0,255,0))]:
            small=cv2.resize(crop_with_padding(mask,bounds),(280,250),interpolation=cv2.INTER_NEAREST)
            contours,_=cv2.findContours(small,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tile,contours,-1,color,1)
        cv2.putText(tile,f'{name[:8]} IoU {iou:.2f}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        tiles.append(tile)
    while len(tiles)%4: tiles.append(np.zeros((250,280,3),np.uint8))
    if tiles: cv2.imwrite(str(ROOT/'error_contact_sheet.png'),np.concatenate([np.concatenate(tiles[i:i+4],1) for i in range(0,len(tiles),4)],0))
    print(json.dumps(json.loads((ROOT/'diagnosis.json').read_text())['oracle_scenarios']),flush=True)


def prepare():
    if (ROOT/'cache_manifest.json').exists(): raise FileExistsError('Cache exists')
    (ROOT/'cache').mkdir(parents=True,exist_ok=True)
    train,val=names(); calibration=set(train[-round(len(train)*.2):])
    source=[json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    meta=json.loads(SOURCE.with_suffix('.meta.json').read_text())
    assert sorted(r['file_name'] for r in source)==train and meta['mode']=='train'
    assert sha(meta['refiner'])==meta['refiner_sha256']
    target_ann=Annotations(Path('data/clean/train_annotations_medoid.json'))
    all_rows=[]; start=time.time()
    for index,item in enumerate(source):
        name=item['file_name']; assert name not in val
        gt=[g for group in target_ann.get(name) for g in group]
        candidates=sorted(item['candidates'],key=lambda c:-c['features'][0])
        kept=[]
        for c in candidates:
            if c['features'][0]<.12: continue
            p=dict(size=item['shape'],counts=c['rle'].encode())
            if kept and matrix([p],[k[0] for k in kept]).max()>.5: continue
            kept.append((p,c))
            if len(kept)==MAX_CANDIDATES: break
        if not kept: continue
        overlap=matrix([k[0] for k in kept],gt)
        image=read_gray_image(config.TRAIN_IMG_DIR/name)
        xs=[]; ys=[]; records=[]
        for j,(p,c) in enumerate(kept):
            initial=coco.decode(p); best=float(overlap[j].max()) if gt else 0.
            target=coco.decode(gt[int(overlap[j].argmax())]) if best>=.1 else initial
            x,y,bounds=features(image,initial,target)
            xs.append(x); ys.append(y.astype(np.uint8))
            records.append(dict(file_name=name,index=j,initial_iou=best,supervision='canonical' if best>=.1 else 'identity',
                                calibration=name in calibration,bounds=list(bounds),rle=p['counts'].decode(),
                                target_rle=encode(target)['counts'].decode(),shape=item['shape']))
        path=ROOT/'cache'/f'{Path(name).stem}.npz'
        np.savez_compressed(path,x=np.stack(xs),y=np.stack(ys))
        for r in records: r['cache']=str(path)
        all_rows.extend(records)
        if (index+1)%50==0: print(f'prepared={index+1}/{len(source)} crops={len(all_rows)} seconds={time.time()-start:.0f}',flush=True)
    save('cache_manifest.json',dict(records=all_rows,train_files=[n for n in train if n not in calibration],
         calibration_files=sorted(calibration),final_validation_files=val,source_sha256=sha(SOURCE),
         target_sha256=sha('data/clean/train_annotations_medoid.json'),upstream_metadata=meta,
         size=SIZE,crop_pad=CROP_PAD,seed=SEED,caveat='Only detector is OOF; frozen refiner saw training images. Training-mask distribution is documented in source metadata.'))
    print(f'cache_complete crops={len(all_rows)}',flush=True)


class Crops(Dataset):
    def __init__(self,records,augment):
        self.records=records; self.augment=augment; self.cache={}
        for path in {r['cache'] for r in records}:
            with np.load(path) as d: self.cache[path]=(d['x'],d['y'])
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; xx,yy=self.cache[r['cache']]
        x=xx[r['index']].astype(np.float32)/255; y=yy[r['index']][None].astype(np.float32)
        if self.augment:
            for dim in [1,2]:
                if random.random()<.5: x=np.flip(x,dim); y=np.flip(y,dim)
        return torch.from_numpy(x.copy()),torch.from_numpy(y.copy())


def utility(pred,y):
    p=pred>0; t=y>.5
    ov=(p&t).sum((1,2,3)).float()/(p|t).sum((1,2,3)).clamp_min(1)
    return torch.where(ov>.5,ov,0.)


def train_model():
    if (ROOT/'corrector.pth').exists(): raise FileExistsError('Model exists')
    manifest=json.loads((ROOT/'cache_manifest.json').read_text())
    if manifest['size']!=SIZE or abs(manifest.get('crop_pad',.15)-CROP_PAD)>1e-8:
        raise ValueError('Cache preprocessing differs from runtime')
    tr=[r for r in manifest['records'] if not r['calibration']]; va=[r for r in manifest['records'] if r['calibration']]
    assert not {r['file_name'] for r in tr}&{r['file_name'] for r in va}
    assert not {r['file_name'] for r in tr+va}&set(manifest['final_validation_files'])
    training=DataLoader(Crops(tr,True),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    validation=DataLoader(Crops(va,False),batch_size=BATCH_SIZE,num_workers=0)
    model=Corrector().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda'); history=[]; best=-float('inf'); start=time.time()
    for epoch in range(17):
        losses=[]
        if epoch:
            model.train()
            for x,y in training:
                x,y=x.cuda(),y.cuda(); optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    delta=model(x); logits=BASE_LOGIT*(2*x[:,2:3]-1)+delta; probs=logits.sigmoid()
                    inter=(probs*y).sum((1,2,3)); union=(probs+y-probs*y).sum((1,2,3))
                    loss=.35*F.binary_cross_entropy_with_logits(logits,y)+.65*(1-(inter+1)/(union+1)).mean()+RESIDUAL_PENALTY*delta.square().mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); losses.append(loss.item())
        model.eval(); values=[]; baseline=[]
        with torch.no_grad():
            for x,y in validation:
                x,y=x.cuda(),y.cuda(); prior=BASE_LOGIT*(2*x[:,2:3]-1)
                values.extend(utility(prior+model(x),y).cpu().tolist())
                baseline.extend(utility(prior,y).cpu().tolist())
        metric=float(np.mean(values)); row=dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,
            calibration_crop_utility=metric,baseline_crop_utility=float(np.mean(baseline)),seconds=time.time()-start)
        history.append(row); save('history.json',history); print(json.dumps(row),flush=True)
        if metric>best:
            best=metric
            torch.save(dict(model=model.state_dict(),epoch=epoch,train_file_names=manifest['train_files'],
                calibration_files=manifest['calibration_files'],final_validation_files=manifest['final_validation_files'],
                metric=metric,seed=SEED,size=SIZE,crop_pad=CROP_PAD,base_logit=BASE_LOGIT,residual_limit=RESIDUAL_LIMIT,
                residual_penalty=RESIDUAL_PENALTY,source_sha256=manifest['source_sha256']),ROOT/'corrector.pth')
    calibrate()


def load_corrector():
    c=torch.load(ROOT/'corrector.pth',map_location='cpu',weights_only=False)
    if c['size']!=SIZE or abs(c.get('crop_pad',.15)-CROP_PAD)>1e-8:
        raise ValueError('Checkpoint preprocessing differs from runtime')
    if c.get('base_logit',1.)!=BASE_LOGIT or c.get('residual_limit',3.)!=RESIDUAL_LIMIT:
        raise ValueError('Checkpoint residual parameterization differs from runtime')
    model=Corrector().cuda().eval(); model.load_state_dict(c['model']); return model,c


@torch.no_grad()
def calibrate():
    model,ckpt=load_corrector(); manifest=json.loads((ROOT/'cache_manifest.json').read_text())
    rows=[r for r in manifest['records'] if r['calibration']]
    totals={s:[] for s in [0.,.5,1.]}
    # Native-resolution single-instance correction applies the exact inference area guard.
    for path in sorted({r['cache'] for r in rows}):
        selected=[r for r in rows if r['cache']==path]
        with np.load(path) as data: x=torch.from_numpy(data['x'].astype(np.float32)/255).cuda()
        delta=model(x).cpu().numpy()[:,0]
        for r in selected:
            p=dict(size=r['shape'],counts=r['rle'].encode()); gt=dict(size=r['shape'],counts=r['target_rle'].encode())
            initial=coco.decode(p)
            for strength in totals:
                fixed=corrected_mask(initial,delta[r['index']],r['bounds'],strength)
                ov=float(matrix([encode(fixed)],[gt])[0,0]); totals[strength].append(ov if ov>.5 else 0.)
    utilities={s:float(np.mean(v)) for s,v in totals.items()}
    strength=max(utilities,key=lambda s:(utilities[s],-s))
    save('calibration.json',dict(strength=strength,utilities=utilities,epoch=ckpt['epoch'],
         calibration_crops=len(rows),warning='Single-instance training-side calibration; full-image promotion is separate.'))


@torch.no_grad()
def correct_image(model,image,masks,strength):
    if not masks or strength==0: return [m.copy() for m in masks]
    inputs=[]; bounds=[]
    for m in masks:
        x,_,b=features(image,m); inputs.append(x); bounds.append(b)
    deltas=[]
    for start in range(0,len(inputs),16):
        batch=torch.from_numpy(np.stack(inputs[start:start+16]).astype(np.float32)/255).cuda()
        deltas.extend(model(batch).cpu().numpy()[:,0])
    proposed=[corrected_mask(m,d,b,strength) for m,d,b in zip(masks,deltas,bounds)]
    return disjoint_corrections(masks,proposed)


def evaluate():
    if (ROOT/'summary.json').exists(): raise FileExistsError('Evaluation exists')
    model,checkpoint=load_corrector(); calibration=json.loads((ROOT/'calibration.json').read_text())
    _,val=names(); assert not set(val)&set(checkpoint['train_file_names']+checkpoint['calibration_files'])
    strength=calibration['strength']; ann=Annotations(config.ANNOTATION_PATH); frame=pd.read_csv(BASE)
    counts={k:[] for k in ['baseline','corrected']}; rows=[]; changes=0
    for index,name in enumerate(val):
        image=read_gray_image(config.TRAIN_IMG_DIR/name); encoded=predictions(frame,name,image.shape)
        original=[coco.decode(p) for p in encoded]; fixed=correct_image(model,image,original,strength)
        assert len(original)==len(fixed)
        changes+=sum(not np.array_equal(a,b) for a,b in zip(original,fixed))
        changed=[encode(m) for m in fixed]; groups=ann.get(name)
        for k,pp in [('baseline',encoded),('corrected',changed)]:
            counts[k].append(sum((terms(pp,g) for g in groups),np.zeros(4)))
        for j,p in enumerate(changed,1): rows.append(dict(file_name=name,filament_id=f'{Path(name).stem}_{j}',segmentation_rle=p['counts'].decode()))
        if (index+1)%30==0: print(f'evaluated={index+1}/{len(val)}',flush=True)
    table=[]; blocks=np.array_split(np.arange(len(val)),4)
    for k,v in counts.items():
        array=np.asarray(v)
        table.append(dict(policy=k,**score(v),**{f'block_{i+1}':score(array[b])['pq'] for i,b in enumerate(blocks)}))
        df=pd.DataFrame(v,columns=['iou_sum','tp','fp','fn']).assign(file_name=val)
        df.to_csv(ROOT/f'{k}_counts.csv',index=False)
        df.assign(year=df.file_name.str[:4]).groupby('year')[['iou_sum','tp','fp','fn']].sum().to_csv(ROOT/f'{k}_by_year.csv')
    rng=np.random.default_rng(SEED); differences=[]
    for _ in range(2000):
        idx=rng.integers(0,len(val),len(val))
        differences.append(score(np.asarray(counts['corrected'])[idx])['pq']-score(np.asarray(counts['baseline'])[idx])['pq'])
    ci=np.quantile(differences,[.025,.975]).tolist(); delta=table[1]['pq']-table[0]['pq']
    improved=sum(table[1][f'block_{i}']>table[0][f'block_{i}'] for i in range(1,5))
    promote=bool(strength>0 and delta>=.002 and ci[0]>0 and improved>=3)
    pd.DataFrame(table).to_csv(ROOT/'comparison.csv',index=False)
    pd.DataFrame(rows,columns=['file_name','filament_id','segmentation_rle']).to_csv(ROOT/'corrected_valpred.csv',index=False)
    result=dict(images=len(val),comparison=table,delta=delta,paired_bootstrap_95=ci,improved_blocks=improved,
        changed_masks=changes,strength=strength,epoch=checkpoint['epoch'],promote=promote,
        checkpoint_sha256=sha(ROOT/'corrector.pth'),baseline_sha256=sha(BASE),
        warning='Historical reused validation; detector-only OOF upstream; leaderboard result unknown.')
    save('summary.json',result); save('run_summary.json',dict(status='evaluated',**result))
    print(json.dumps(result),flush=True)


def export():
    result=json.loads((ROOT/'summary.json').read_text())
    if not result['promote']:
        print('NOT PROMOTED: no new submission generated',flush=True); return
    path=Path('submissions/submission_obb_gate_extras_pq04502_20260901.csv')
    output=EXPORT_PATH
    if output.exists(): raise FileExistsError(output)
    model,_=load_corrector(); base=pd.read_csv(path); ids=base.filament_id.str.rsplit('_',n=1).str[0]
    rows=[]; changed=0
    for p in sorted(config.TEST_IMG_DIR.iterdir()):
        if p.suffix.lower() not in {'.jpeg','.jpg','.png'}: continue
        image=read_gray_image(p); rr=base.loc[ids==p.stem]
        masks=[coco.decode(dict(size=list(image.shape),counts=s.encode())) for s in rr.segmentation_rle]
        fixed=correct_image(model,image,masks,result['strength'])
        changed+=sum(not np.array_equal(a,b) for a,b in zip(masks,fixed))
        for fid,m in zip(rr.filament_id,fixed): rows.append(dict(filament_id=fid,segmentation_rle=encode(m)['counts'].decode()))
    frame=pd.DataFrame(rows); assert len(frame)==len(base) and not frame.filament_id.duplicated().any()
    frame.to_csv(output,index=False)
    save('submission_audit.json',dict(path=str(output),sha256=sha(output),baseline_sha256=sha(path),rows=len(frame),
         changed_masks=changed,no_overlaps=True,leaderboard_score=None,uploaded=False))


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['audit','prepare','train','calibrate','evaluate','export']); a=p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(2); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    ROOT.mkdir(parents=True,exist_ok=True)
    {'audit':audit,'prepare':prepare,'train':train_model,'calibrate':calibrate,'evaluate':evaluate,'export':export}[a.stage]()


if __name__=='__main__': main()
