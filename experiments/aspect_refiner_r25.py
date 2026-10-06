"""Fixed-budget geometry pilot. Frozen historical final split; no LB selection."""
import argparse
import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import config
from experiments import mask_correction_r17 as core
from experiments.evaluate_hbb_adaptation import load_model
from experiments.evaluate_proposal_adaptation import decode
from src.crop_refiner_utils import (crop_bounds_from_bbox, crop_bounds_from_bbox_rect,
    crop_with_padding, make_refiner_channels, load_instance_records, decode_consensus_mask, seed_from_bbox)
from src.dataset import read_gray_image
from src.losses import build_loss
from src.assemble_obb_gate_submission import assemble_image
from src.frozen_mask_quality import Annotations, encode, score, terms
from src.masks import postprocess_binary_mask
from src.predict_detect_refine_multiscale import mask_nms_scored, make_masks_disjoint_scored

ROOT = Path('experiments/aspect_refiner_r25_20260922')
INIT = Path('models/crop_refiner_resnet34_512_ioubce_ft.pth')
SHAPE = Path('models/crop_refiner_resnet34_512_consensus_ft.pth')
SOURCE = Path('data/clean/hbb_oof_r4_proposals.json')
SEED = 20260922
SAMPLES = 512
EPOCHS = 2


def save(name, data):
    (ROOT/name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def geometry(bbox, shape, mode):
    """Use rectangular 768x384 only for elongated proposals, with isotropic resize."""
    ratio = max(bbox[2:]) / max(1., min(bbox[2:]))
    rect = mode == 'aspect' and ratio >= 2.
    fn = crop_bounds_from_bbox_rect if rect else crop_bounds_from_bbox
    bounds = fn(bbox, *shape, pad=.5)
    h, w = bounds[3]-bounds[1], bounds[2]-bounds[0]
    rotated = rect and h > w
    rh, rw = (w, h) if rotated else (h, w)
    oh, ow = (384, 768) if rect else (512, 512)
    scale = min(oh/rh, ow/rw)
    nh, nw = max(1, round(rh*scale)), max(1, round(rw*scale))
    if not rect:
        nh,nw=oh,ow
    return dict(bounds=bounds, rotated=rotated, resized=(nh,nw), canvas=(oh,ow),
                offset=((oh-nh)//2,(ow-nw)//2), scale=scale, elongated=rect)


def project(array, meta, interpolation=cv2.INTER_LINEAR):
    cropped = crop_with_padding(array, meta['bounds'])
    if meta['rotated']:
        cropped = np.rot90(cropped)
    nh,nw = meta['resized']; y,x = meta['offset']
    output = np.zeros(meta['canvas'], np.float32)
    output[y:y+nh,x:x+nw] = cv2.resize(cropped,(nw,nh),interpolation=interpolation)
    return output


def restore(array, meta):
    nh,nw = meta['resized']; y,x = meta['offset']
    cropped = array[y:y+nh,x:x+nw]
    x0,y0,x1,y1 = meta['bounds']
    size = (y1-y0,x1-x0) if meta['rotated'] else (x1-x0,y1-y0)
    result = cv2.resize(cropped,size,interpolation=cv2.INTER_LINEAR)
    return np.rot90(result,-1).copy() if meta['rotated'] else result


def inputs(image,bbox,mode):
    meta = geometry(bbox,image.shape,mode)
    # Native-coordinate prompt, projected through exactly the image transform.
    seed = np.zeros(image.shape,np.float32)
    x,y,w,h = bbox
    x0,y0 = max(0,round(x)),max(0,round(y))
    x1,y1 = min(image.shape[1],round(x+w)),min(image.shape[0],round(y+h))
    if x1>x0 and y1>y0:
        seed[y0:y1,x0:x1]=1
    crop = project(image,meta)
    prompt = project(seed,meta,cv2.INTER_NEAREST)
    if not meta['elongated']:
        prompt = seed_from_bbox(bbox,meta['bounds'],512)
    return make_refiner_channels(crop,prompt),meta


def register():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir()
    train,val=core.names()
    save('preregistration.json',dict(timestamp=datetime.datetime.now().isoformat(),
        hypothesis='Matched train/inference isotropic elongated crops recover detail lost to square-context downsampling.',
        arms=['square_control','aspect'],epochs=EPOCHS,samples_per_epoch=SAMPLES,
        seed=SEED,decoder_lr=3e-5,layer4_lr=3e-6,accumulation=4,
        geometry='aspect>=2 uses 768x384 letterbox; all other proposals use 512x512 square',
        sampling='60% detector-only OOF prompts, 40% consensus GT; same draws/augmentation in both arms; elongation weight sqrt(ratio) capped at 3',
        init_sha256=core.sha(INIT),shape_sha256=core.sha(SHAPE),source_sha256=core.sha(SOURCE),
        train_files=train,validation_files=val,selection='fixed final epoch; no validation threshold/epoch tuning',
        gate=dict(min_delta_vs_r22=.002,bootstrap_lower_gt=0,min_improved_blocks=3),
        limitations='Short pilot, not convergence test. Approximate equal compute: rectangular canvas has 12.5% more pixels. Refiners have seen train pool; detector-only OOF is not full-pipeline OOF. Historical validation reused.'))


def train_arm(mode):
    out=ROOT/f'{mode}.pth'
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng=np.random.default_rng(SEED)
    train,val=core.names()
    proposals=json.loads(SOURCE.read_text())['proposals']
    assert all(r['file_name'] in set(train) for r in proposals)
    _,gt=load_instance_records(Path('data/clean/train_annotations_consensus.json'),set(train))
    pools=[proposals,gt]; weights=[]
    for pool in pools:
        ratios=[max(r['bbox'][2:])/max(1.,min(r['bbox'][2:])) for r in pool]
        w=np.minimum(3,np.sqrt(ratios)); weights.append(w/w.sum())
    model,ckpt=load_model(INIT)
    for n,p in model.named_parameters():
        p.requires_grad_(not n.startswith('encoder.') or n.startswith('encoder.layer4.'))
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters(): p.requires_grad_(False)
    frozen={n:v.cpu().clone() for n,v in model.state_dict().items()
            if n.startswith('encoder.') and not n.startswith('encoder.layer4.')}
    opt=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')],lr=3e-5),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')],lr=3e-6)],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda'); criterion=build_loss('iou_bce')
    history=[]; start=time.time(); elongated=0
    for epoch in range(EPOCHS):
        model.train()
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm): m.eval()
        opt.zero_grad(set_to_none=True); losses=[]
        for i in range(SAMPLES):
            pool_id=0 if rng.random()<.6 else 1
            record=pools[pool_id][rng.choice(len(pools[pool_id]),p=weights[pool_id])]
            image=read_gray_image(config.TRAIN_IMG_DIR/record['file_name'])
            bbox=record.get('proposal_bbox',record['bbox'])
            x,meta=inputs(image,bbox,mode)
            target=project(decode_consensus_mask(record),meta)
            elongated+=int(meta['elongated'])
            for axis in [1,2]:
                if rng.random()<.5:
                    x=np.flip(x,axis); target=np.flip(target,axis-1)
            xt=torch.from_numpy(x.copy()[None]).cuda(); yt=torch.from_numpy(target.copy()[None,None]).cuda()
            with torch.amp.autocast('cuda'):
                loss=criterion(model(xt),yt)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            scaler.scale(loss/4).backward(); losses.append(float(loss))
            if (i+1)%4==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if (i+1)%128==0:
                print(mode,epoch+1,i+1,round(np.mean(losses),5),round(time.time()-start),flush=True)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-start))
        save(f'{mode}_history.json',history)
    assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
    ckpt.update(model=model.state_dict(),r25_geometry=mode,train_file_names=train,final_validation_files=val,
                epoch=EPOCHS,elongated_training_draws=elongated)
    torch.save(ckpt,out)
    del model,opt,ckpt
    torch.cuda.empty_cache()


@torch.no_grad()
def probabilities(model,x):
    batch=torch.from_numpy(x.copy()[None]).cuda()
    result=[]
    for axes in [(),(3,),(2,),(2,3)]:
        transformed=batch.flip(axes) if axes else batch
        p=model(transformed).sigmoid()
        if axes:p=p.flip(axes)
        result.append(p[0,0].cpu().numpy())
    return np.mean(result,axis=0)


def paste(prob,meta,image_shape):
    binary=postprocess_binary_mask((prob>.45).astype(np.uint8),'close')
    x0,y0,x1,y1=meta['bounds']; h,w=image_shape
    sx0,sy0,sx1,sy1=max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    mask=np.zeros(image_shape,np.uint8)
    if sx1>sx0 and sy1>sy0:
        mask[sy0:sy1,sx0:sx1]=binary[sy0-y0:sy1-y0,sx0-x0:sx1-x0]
    return mask


@torch.no_grad()
def evaluate():
    if (ROOT/'summary.json').exists(): raise FileExistsError('Already evaluated')
    models={k:load_model(p)[0] for k,p in [('original',INIT),('square_control',ROOT/'square_control.pth'),('aspect',ROOT/'aspect.pth')]}
    shape,_=load_model(SHAPE)
    core.ROOT=Path('experiments/mask_correction_r22_20260918')
    corrector,_=core.load_corrector()
    _,val=core.names(); ann=Annotations(config.ANNOTATION_PATH)
    raw={r['file_name']:r for r in json.loads(Path('experiments/pipeline_diagnosis_r5/detector_boxes.json').read_text())}
    confirm=pd.read_csv('experiments/valpred_obb_refine_tta_c030.csv')
    extras=pd.read_csv('experiments/valpred_obb_refine_tta_c038.csv')
    opts=argparse.Namespace(confirm_iou=.1,unmatched_min_area=3000,extra_match_iou=.1,extra_min_area=500)
    counts={k:[] for k in models}; rows={k:[] for k in models}; start=time.time()
    for i,name in enumerate(val):
        image=read_gray_image(config.TRAIN_IMG_DIR/name); candidates={k:[] for k in models}
        for box,s in zip(raw[name]['boxes'],raw[name]['scores']):
            if s<.22:continue
            x0,y0,x1,y1=box; bbox=(x0,y0,x1-x0,y1-y0)
            square,ms=inputs(image,bbox,'square_control')
            shape_prob=restore(probabilities(shape,square),ms)
            for k,model in models.items():
                x,meta=inputs(image,bbox,k) if k=='aspect' else (square,ms)
                primary=restore(probabilities(model,x),meta)
                # Shape checkpoint remains on its native square preprocessing.
                shape_full=np.zeros(image.shape,np.float32)
                a,b,c,d=ms['bounds']; sa,sb,sc,sd=max(0,a),max(0,b),min(image.shape[1],c),min(image.shape[0],d)
                if sc>sa and sd>sb:shape_full[sb:sd,sa:sc]=shape_prob[sb-b:sd-b,sa-a:sc-a]
                if int((primary>.45).sum())<120:continue
                mixed=.8*primary+.2*crop_with_padding(shape_full,meta['bounds'])
                mask=paste(mixed,meta,image.shape)
                if mask.sum()>=120:candidates[k].append((mask,float(s)))
        for k in models:
            hbb=[m for m,s in make_masks_disjoint_scored(mask_nms_scored(candidates[k],.12),120)]
            assembled=assemble_image(hbb,decode(confirm,name,image.shape),decode(extras,name,image.shape),opts)[0]
            fixed=core.correct_image(corrector,image,assembled,1.)
            encoded=[encode(m) for m in fixed]
            counts[k].append(sum((terms(encoded,g) for g in ann.get(name)),np.zeros(4)))
            for j,m in enumerate(encoded,1):rows[k].append(dict(file_name=name,filament_id=f'{Path(name).stem}_{j}',segmentation_rle=m['counts'].decode()))
        if (i+1)%10==0:print('evaluation',i+1,len(val),'seconds',round(time.time()-start),flush=True)
    table=[]
    for k,v in counts.items():
        a=np.asarray(v)
        pd.DataFrame(a,columns=['iou_sum','tp','fp','fn']).assign(file_name=val).to_csv(ROOT/f'{k}_counts.csv',index=False)
        pd.DataFrame(rows[k],columns=['file_name','filament_id','segmentation_rle']).to_csv(ROOT/f'{k}_valpred.csv',index=False)
        table.append(dict(policy=k,**score(a),**{f'block_{j+1}':score(a[idx])['pq'] for j,idx in enumerate(np.array_split(np.arange(len(val)),4))}))
    comparisons={}
    for base in ['original','square_control']:
        rng=np.random.default_rng(SEED); differences=[]
        for _ in range(2000):
            idx=rng.integers(0,len(val),len(val))
            differences.append(score(np.asarray(counts['aspect'])[idx])['pq']-score(np.asarray(counts[base])[idx])['pq'])
        comparisons[base]=dict(delta=score(counts['aspect'])['pq']-score(counts[base])['pq'],ci95=np.quantile(differences,[.025,.975]).tolist())
    improved=sum(table[2][f'block_{j}']>table[0][f'block_{j}'] for j in range(1,5))
    reference=json.loads((core.ROOT/'summary.json').read_text())['comparison'][1]['pq']
    reproduction=abs(table[0]['pq']-reference)<1e-5
    promote=bool(reproduction and comparisons['original']['delta']>=.002 and comparisons['original']['ci95'][0]>0 and improved>=3)
    save('summary.json',dict(comparison=table,aspect_vs=comparisons,improved_blocks=improved,
        r22_reference_reproduced=reproduction,promote=promote,warning='Fixed short pilot on historically reused validation. No Kaggle score.'))
    pd.DataFrame(table).to_csv(ROOT/'comparison.csv',index=False)
    print(json.dumps(json.loads((ROOT/'summary.json').read_text())),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','evaluate']);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    if a.stage=='run':
        register();train_arm('square_control');train_arm('aspect')
    evaluate()
    assert core.sha(INIT)==json.loads((ROOT/'preregistration.json').read_text())['init_sha256']


if __name__=='__main__':main()
