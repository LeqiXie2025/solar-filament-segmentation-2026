"""Frozen paired continuation for R25 square primary; only polygon target rasterizer differs."""
import os,sys,json,hashlib,time,random
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
from pycocotools import mask as coco
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'experiments/route_recovery_20261004/r25_proposal/repair_01';sys.path.insert(0,str(ROOT))
from src.crop_refiner_utils import load_instance_records,crop_with_padding,make_refiner_channels
from src.dataset import read_gray_image
from src.losses import build_loss
from src.model_factory import create_model_from_checkpoint
from experiments.evaluate_hbb_adaptation import load_model
import experiments.aspect_refiner_r25 as r25

PRE=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
SEED=20260922;SIZE=512;BATCH=1;ACCUM=4

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)
def rng_state_bytes(rng):return rng.bit_generator.state

def decode_poly_coco(polygons,h,w):
 if not polygons:return np.zeros((h,w),np.uint8)
 rs=coco.frPyObjects(polygons,h,w);merged=coco.merge(rs) if rs else None
 return coco.decode(merged).astype(np.uint8) if merged is not None else np.zeros((h,w),np.uint8)
def decode_target(record,arm):
 """Same member selection/averaging and RLE fallback; only polygon rasterizer swaps."""
 h,w=int(record['height']),int(record['width']);cons=record.get('consensus_segmentations') or []
 def one(seg):
  if isinstance(seg,list):
   if arm=='control_cv2':return r25.decode_consensus_mask({**record,'segmentation':seg,'consensus_segmentations':[]}).astype(np.float32)
   return decode_poly_coco(seg,h,w).astype(np.float32)
  if isinstance(seg,dict):return r25.decode_consensus_mask({**record,'segmentation':seg,'consensus_segmentations':[]}).astype(np.float32)
  return np.zeros((h,w),np.float32)
 if len(cons)<=1:return one(record.get('segmentation'))
 masks=[one(seg) for seg in cons if isinstance(seg,(list,dict))]
 return np.mean(np.stack(masks),axis=0).astype(np.float32) if masks else one(record.get('segmentation'))

def train_arm(arm,pools,draws):
 final=OUT/f'{arm}_epoch6.pth';resume=OUT/f'{arm}_resume.pth'
 if final.exists():raise FileExistsError(final)
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.set_num_threads(4);cv2.setNumThreads(2)
 model,base=load_model(ROOT/'experiments/aspect_refiner_r25_20260922/square_control.pth')
 # The base wrapper returns the frozen square checkpoint on CUDA; reset all states from identical source.
 for n,p in model.named_parameters():p.requires_grad_(not n.startswith('encoder.') or n.startswith('encoder.layer4.'))
 for m in model.modules():
  if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
   for p in m.parameters():p.requires_grad_(False)
 frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n.startswith('encoder.') and not n.startswith('encoder.layer4.')}
 opt=torch.optim.AdamW([
  {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')],'lr':3e-5},
  {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')],'lr':3e-6}],weight_decay=1e-4)
 scaler=torch.amp.GradScaler('cuda');criterion=build_loss('iou_bce');augment_rng=np.random.default_rng(SEED+1)
 if resume.exists():
  ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['arm']==arm and ck['schedule_sha256']==PRE['sampler_schedule_sha256']
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);augment_rng.bit_generator.state=ck['augment_rng_state'];torch.set_rng_state(ck['torch_rng_state']);torch.cuda.set_rng_state_all(ck['cuda_rng_state']);start_epoch=int(ck['epoch_completed'])
 else:start_epoch=0
 model.train();start=time.time()
 cache={};cache_bytes=0
 for ep in range(start_epoch,6):
  for mod in model.modules():
   if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
  opt.zero_grad(set_to_none=True);losses=[];epoch_draws=draws[ep*512:(ep+1)*512]
  for local,(pool_id,index) in enumerate(epoch_draws):
   rec=pools[pool_id][index];name=rec['file_name']
   if name not in cache:
    image=read_gray_image(ROOT/'data/raw/train_images'/name);cache[name]=image;cache_bytes+=image.nbytes
   image=cache[name];bbox=rec.get('proposal_bbox',rec['bbox']);x,meta=r25.inputs(image,bbox,'square_control')
   target=decode_target(rec,arm);target=r25.project(target,meta)
   if augment_rng.random()<.5:x=np.flip(x,axis=1);target=np.flip(target,axis=0)
   if augment_rng.random()<.5:x=np.flip(x,axis=2);target=np.flip(target,axis=1)
   xt=torch.from_numpy(np.ascontiguousarray(x[None])).float().cuda();yt=torch.from_numpy(np.ascontiguousarray(target[None,None])).float().cuda()
   with torch.amp.autocast('cuda'):loss=criterion(model(xt),yt)
   if not torch.isfinite(loss):raise RuntimeError(f'nonfinite {arm} epoch {ep+1} sample {local}')
   scaler.scale(loss/ACCUM).backward();losses.append(float(loss.detach()))
   if (local+1)%ACCUM==0:
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
   if (local+1)%128==0:print(json.dumps({'stage':'train','arm':arm,'epoch':ep+1,'samples':local+1,'loss_mean':float(np.mean(losses))}),flush=True)
  assert all(torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in frozen.items()),'frozen encoder changed'
  payload={'model':model.state_dict(),'arch':base['arch'],'encoder_name':base['encoder_name'],'encoder_weights':base.get('encoder_weights'),'in_channels':3,'image_size':512,'threshold':base.get('threshold',.45),'mode':'r25_square_target_rasterizer','epoch':ep+1,'init_sha256':PRE['sources']['square_primary_sha256'],'train_file_names':PRE['split']['train_files'],'final_validation_files':PRE['split']['validation_files'],'arm':arm,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'augment_rng_state':augment_rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'epoch_completed':ep+1,'draw_position':(ep+1)*512,'schedule_sha256':PRE['sampler_schedule_sha256'],'optimizer_config':PRE['initialization_and_optimization']}
  tmp=Path(str(resume)+'.tmp');torch.save(payload,tmp);tmp.replace(resume)
  atomic_json(OUT/'state.json',{'phase':'training','pid':os.getpid(),'arm':arm,'epoch_completed':ep+1,'draw_position':(ep+1)*512,'schedule_sha256':PRE['sampler_schedule_sha256'],'updated_local':time.strftime('%Y-%m-%dT%H:%M:%S')})
  with (OUT/f'{arm}_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'epoch':ep+1,'mean_loss':float(np.mean(losses)),'seconds_elapsed':time.time()-start,'image_cache_bytes':cache_bytes})+'\n')
 # Retain complete optimizer state in final checkpoint; don't overwrite original source.
 final_tmp=Path(str(final)+'.tmp');final_tmp.write_bytes(resume.read_bytes());final_tmp.replace(final)
 del model,opt;torch.cuda.empty_cache()

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
 assert PRE['status']=='preregistered_before_training'
 checks=PRE['sources']
 for label,path in checks['path_by_key'].items():
  assert sha(ROOT/path)==checks['sha256_by_key'][label],f'source changed: {path}'
 schedule=json.loads((OUT/'sampler_schedule.json').read_text());assert hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest()==PRE['sampler_schedule_sha256']
 fit=set(PRE['split']['train_files']);baseck=torch.load(ROOT/'experiments/aspect_refiner_r25_20260922/square_control.pth',map_location='cpu',weights_only=False)
 assert set(baseck['train_file_names'])==fit and not fit&set(PRE['split']['validation_files'])
 src=json.loads((ROOT/'data/clean/hbb_oof_r4_proposals.json').read_text());props=[r for r in src['proposals'] if r['file_name'] in fit];_,gts=load_instance_records(ROOT/'data/clean/train_annotations_consensus.json',fit);pools=[props,gts]
 draws=[(0 if d['pool']=='oof_prompt' else 1,int(d['pool_index'])) for d in schedule['draws']]
 assert len(draws)==3072 and all((pid==0 and idx<len(props)) or (pid==1 and idx<len(gts)) for pid,idx in draws)
 atomic_json(OUT/'state.json',{'phase':'training_start','pid':os.getpid(),'candidate_and_control':'sequential single GPU','train_images':len(fit),'validation_images':len(PRE['split']['validation_files']),'schedule_sha256':PRE['sampler_schedule_sha256'],'started_local':time.strftime('%Y-%m-%dT%H:%M:%S')})
 train_arm('control_cv2',pools,draws);train_arm('candidate_coco',pools,draws)
 atomic_json(OUT/'state.json',{'phase':'training_complete','pid':None,'control_checkpoint_sha256':sha(OUT/'control_cv2_epoch6.pth'),'candidate_checkpoint_sha256':sha(OUT/'candidate_coco_epoch6.pth'),'schedule_sha256':PRE['sampler_schedule_sha256'],'next_action':'Run frozen R25 validation comparison; export CSV only if the preregistered gate passes.'})
 print(json.dumps({'status':'training_complete','control':sha(OUT/'control_cv2_epoch6.pth'),'candidate':sha(OUT/'candidate_coco_epoch6.pth')}),flush=True)
if __name__=='__main__':main()



