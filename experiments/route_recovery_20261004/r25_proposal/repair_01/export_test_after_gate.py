"""Resumable, gated R25 test inference with source/split/RLE QA; never uploads."""
import hashlib,json,os,shutil,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd,torch,config
from pycocotools import mask as coco
from experiments import mask_correction_r17 as core
from experiments.evaluate_hbb_adaptation import load_model
from experiments.evaluate_proposal_adaptation import decode
from src.assemble_obb_gate_submission import assemble_image,decode_submission_rows
from src.dataset import read_gray_image
from src.frozen_mask_quality import encode
from src.predict_detect_refine import patch_torchvision_nms
from src.predict_detect_refine_multiscale import refine_boxes_tta,mask_nms_scored,make_masks_disjoint_scored

OUT=ROOT/'experiments/route_recovery_20261004/r25_proposal/repair_01'
PRE_PATH=OUT/'preregistration.json'
GATE_PATH=OUT/'validation_gate.json'
CACHE_PATH=OUT/'test_inference_checkpoint.jsonl'
STATE_PATH=OUT/'state.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def tensor_fingerprint(state):
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        h.update(key.encode('utf-8')+b'\0'+str(value.dtype).encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0')
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def write_json_atomic(path,obj):
    path=Path(path);tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)

def append_durable(path,obj):
    with Path(path).open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(obj,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def main():
    pre=json.loads(PRE_PATH.read_text(encoding='utf-8'))
    gate=json.loads(GATE_PATH.read_text(encoding='utf-8'))
    required_gates=('candidate_control_delta_min','candidate_control_ci_lower_positive','candidate_tp_at_least_frozen_original','candidate_fp_within_5pct_original','candidate_pq_at_least_original')
    if not gate.get('promote_to_test') or not all(gate.get('gates',{}).get(k,False) for k in required_gates):
        raise RuntimeError('Repaired R25 validation gate is not fully passed; test export forbidden')

    for key,rel in pre['sources']['path_by_key'].items():
        actual=sha(ROOT/rel)
        if actual!=pre['sources']['sha256_by_key'][key]:raise RuntimeError(f'preregistered source changed: {key}')
    manifest_path=ROOT/pre['test_output']['test_image_manifest_file']
    manifest_byte_hash=sha(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest!=pre['test_output']['test_images']:
        raise RuntimeError('Test image manifest content differs from the full manifest embedded in preregistration')
    expected=manifest['images'];test_names=[r['file_name'] for r in expected]
    if int(manifest['count'])!=180 or len(expected)!=180 or len(set(test_names))!=180:
        raise RuntimeError('Test manifest must contain exactly 180 unique images')
    val_names=set(pre['split']['validation_files']);fit_names=set(pre['split']['train_files']);test_set=set(test_names)
    if len(val_names)!=141 or test_set&val_names or test_set&fit_names:
        raise RuntimeError('Test IDs overlap frozen fit/validation IDs or split cardinality changed')

    files={p.name:p for p in config.TEST_IMG_DIR.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    if sorted(files)!=sorted(test_names):raise RuntimeError('On-disk test image set differs from preregistered manifest')
    test_hashes={};test_bytes={}
    for item in expected:
        path=files[item['file_name']];actual=sha(path);size=path.stat().st_size
        if actual!=item['sha256'] or size!=int(item['size_bytes']):raise RuntimeError(f'Test image changed: {path.name}')
        test_hashes[path.name]=actual;test_bytes[path.name]=size
    val_hashes={}
    for name in sorted(val_names):
        path=config.TRAIN_IMG_DIR/name
        if not path.is_file():raise RuntimeError(f'Missing frozen validation image: {name}')
        val_hashes[name]=sha(path)
    shared_hashes=set(test_hashes.values())&set(val_hashes.values())
    if shared_hashes:raise RuntimeError('Test and validation images share identical file hashes')

    ckpt_path=OUT/'candidate_coco_epoch6.pth';expected_ckpt_hash=gate['source_hashes'].get(str(ckpt_path.relative_to(ROOT)))
    if not expected_ckpt_hash or sha(ckpt_path)!=expected_ckpt_hash:
        raise RuntimeError('Candidate checkpoint SHA does not match the repaired validation gate')
    checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    if checkpoint.get('arm')!='candidate_coco' or checkpoint.get('epoch_completed')!=6:
        raise RuntimeError('Candidate checkpoint metadata is not the repaired six-epoch COCO arm')
    if set(checkpoint.get('train_file_names',[]))!=fit_names:
        raise RuntimeError('Candidate checkpoint fit IDs differ from the frozen preregistration')
    if set(checkpoint.get('train_file_names',[]))&test_set or set(checkpoint.get('final_validation_files',[]))!=val_names:
        raise RuntimeError('Candidate checkpoint train/validation split metadata is inconsistent')
    checkpoint_tensor_hash=tensor_fingerprint(checkpoint['model'])

    proposed=ROOT/pre['test_output']['csv'];download_dir=Path.home()/'Downloads'
    download_dir.mkdir(parents=True,exist_ok=True)
    stem=proposed.stem;ext=proposed.suffix
    output=proposed;download=download_dir/(stem+ext);version=1
    while output.exists() or download.exists():
        version+=1;output=proposed.with_name(f'{stem}_v{version}{ext}');download=download_dir/output.name
    cache_rows={};cache_empty=set();completed=[]
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding='utf-8').splitlines():
            item=json.loads(line);name=item['file_name']
            if name not in test_set or name in cache_rows:raise RuntimeError('Invalid/duplicate durable test inference record')
            cache_rows[name]=item['rows'];completed.append(name)
            if item.get('empty',not bool(item['rows'])):cache_empty.add(name)
        if completed!=test_names[:len(completed)]:raise RuntimeError('Durable test checkpoint is not an exact manifest prefix')

    preflight={'status':'passed_before_test_inference','gate_path':str(GATE_PATH.relative_to(ROOT)),'gate_source':'repair_01 validation only','validation_images':len(val_names),'test_images':len(test_names),'test_ids_disjoint_from_fit_and_validation':True,'test_vs_validation_identical_hashes':0,'test_image_hashes':test_hashes,'test_image_bytes':test_bytes,'validation_image_hashes':val_hashes,'test_manifest_path':str(manifest_path.relative_to(ROOT)),'test_manifest_registered_sha256':pre['test_output']['test_image_manifest_sha256'],'test_manifest_actual_sha256':manifest_byte_hash,'test_manifest_full_semantic_content_matches_embedded_preregistration':True,'candidate_checkpoint_sha256':expected_ckpt_hash,'candidate_checkpoint_tensor_fingerprint':checkpoint_tensor_hash,'checkpoint_arm':checkpoint['arm'],'checkpoint_epoch':checkpoint['epoch_completed'],'fit_images':len(fit_names),'output_csv':str(output.relative_to(ROOT)),'downloads_csv':str(download),'resumed_images':len(completed),'upload':False}
    write_json_atomic(OUT/'test_export_preflight.json',preflight)
    print(json.dumps({'phase':'preflight_complete','test_images':180,'validation_images':141,'candidate_checkpoint_sha256':expected_ckpt_hash,'candidate_tensor_fingerprint':checkpoint_tensor_hash,'test_validation_content_hash_overlap':0,'resumed_images':len(completed)}),flush=True)
    write_json_atomic(STATE_PATH,{'phase':'test_inference_running','pid':os.getpid(),'candidate_checkpoint_sha256':expected_ckpt_hash,'candidate_tensor_fingerprint':checkpoint_tensor_hash,'test_images_total':180,'test_images_completed':len(completed),'checkpoint_file':str(CACHE_PATH.relative_to(ROOT)),'csv_output':str(output.relative_to(ROOT)),'downloads_output':str(download),'started_local':time.strftime('%Y-%m-%dT%H:%M:%S')})

    torch.backends.cudnn.benchmark=False;torch.set_num_threads(4);cv2.setNumThreads(2)
    core.ROOT=ROOT/'experiments/mask_correction_r22_20260918'
    model,loaded=load_model(ckpt_path)
    loaded_tensor_hash=tensor_fingerprint(model.state_dict())
    if loaded_tensor_hash!=checkpoint_tensor_hash:raise RuntimeError('Actual loaded candidate tensor fingerprint differs from checkpoint payload')
    if loaded.get('arm')!='candidate_coco' or sha(ckpt_path)!=expected_ckpt_hash:raise RuntimeError('Loaded candidate identity changed after preflight')
    preflight['loaded_candidate_tensor_fingerprint']=loaded_tensor_hash
    preflight['loaded_tensor_fingerprint_matches_checkpoint']=True
    write_json_atomic(OUT/'test_export_preflight.json',preflight)
    print(json.dumps({'phase':'candidate_loaded_identity_verified','candidate_tensor_fingerprint':loaded_tensor_hash}),flush=True)
    shape,_=load_model(ROOT/'models/crop_refiner_resnet34_512_consensus_ft.pth')
    corrector,_=core.load_corrector()
    opts=type('Opts',(),dict(box_pad=.5,batch_size=12,shape_weight=.2,threshold=.45,min_area=120,score_mode='detector',confirm_iou=.1,unmatched_min_area=3000,extra_match_iou=.1,extra_min_area=500))()
    patch_torchvision_nms()
    from ultralytics import YOLO
    detector=YOLO(str(ROOT/'models/yolo/yolo11s_boxes_1280_e25/weights/best.pt'))
    confirm=pd.read_csv(ROOT/'submissions/submission_obb_refine_tta_c030.csv')
    extras=pd.read_csv(ROOT/'submissions/submission_obb_refine_tta_c038.csv')
    rows_by_image={name:cache_rows.get(name,[]) for name in test_names};empty=sorted(cache_empty)
    start=time.time()
    for i,name in enumerate(test_names):
        if name in cache_rows:continue
        path=files[name];image=read_gray_image(path);boxes=[];scores=[]
        for size in (1024,1280):
            det=detector.predict(str(path),imgsz=size,conf=.22,iou=.45,device=0,verbose=False)[0]
            boxes.extend(det.boxes.xyxy.cpu().numpy());scores.extend(det.boxes.conf.cpu().numpy())
        confirm_masks=decode_submission_rows(confirm,path.stem,image.shape)
        extra_masks=decode_submission_rows(extras,path.stem,image.shape)
        if boxes:
            masks=refine_boxes_tta(model,image,np.asarray(boxes).reshape(-1,4),np.asarray(scores),'cuda',opts,512,shape)
            hbb=[m for m,s in make_masks_disjoint_scored(mask_nms_scored(masks,.12),120)]
        else:hbb=[]
        assembled=assemble_image(hbb,confirm_masks,extra_masks,opts)[0]
        fixed=core.correct_image(corrector,image,assembled,1.)
        occupied=np.zeros(image.shape,dtype=np.uint8);image_rows=[]
        for j,mask in enumerate(fixed,1):
            mask=np.asarray(mask,dtype=np.uint8)
            if mask.shape!=image.shape or not int(mask.sum()):raise RuntimeError(f'Invalid/empty mask in {name}')
            if np.logical_and(occupied,mask).any():raise RuntimeError(f'Overlapping masks after assembly: {name}')
            occupied|=mask
            image_rows.append({'filament_id':f'{path.stem}_{j}','segmentation_rle':encode(mask)['counts'].decode()})
        append_durable(CACHE_PATH,{'file_name':name,'rows':image_rows,'empty':not bool(image_rows),'image_sha256':test_hashes[name]})
        rows_by_image[name]=image_rows
        if not image_rows:empty.append(name)
        write_json_atomic(STATE_PATH,{'phase':'test_inference_running','pid':os.getpid(),'candidate_checkpoint_sha256':expected_ckpt_hash,'candidate_tensor_fingerprint':loaded_tensor_hash,'test_images_total':180,'test_images_completed':i+1,'last_completed_image':name,'checkpoint_file':str(CACHE_PATH.relative_to(ROOT)),'csv_output':str(output.relative_to(ROOT)),'downloads_output':str(download),'elapsed_seconds':round(time.time()-start,1),'updated_local':time.strftime('%Y-%m-%dT%H:%M:%S')})
        if (i+1)%30==0:print(json.dumps({'phase':'test_inference','images':i+1,'total':180,'elapsed_seconds':round(time.time()-start)}),flush=True)

    ordered_rows=[row for name in test_names for row in rows_by_image[name]]
    frame=pd.DataFrame(ordered_rows,columns=['filament_id','segmentation_rle'])
    if frame.empty or frame.filament_id.duplicated().any() or frame.isna().any().any():raise RuntimeError('Submission has empty rows, duplicate IDs, or missing cells')
    if list(frame.columns)!=['filament_id','segmentation_rle']:raise RuntimeError('Submission schema mismatch')
    if len(rows_by_image)!=180 or sorted(rows_by_image)!=sorted(test_names):raise RuntimeError('Not all test images are accounted for')
    decoded_by_image={};rle_checks=0;mask_areas=[]
    for name in test_names:
        image_rows=rows_by_image[name];decoded=[]
        for row in image_rows:
            encoded={'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')}
            mask=coco.decode(encoded).astype(np.uint8)
            if mask.shape!=(2048,2048) or int(mask.sum())<=0:raise RuntimeError(f'Bad/empty 2048 RLE: {name}')
            if decoded and any(np.logical_and(mask,other).any() for other in decoded):raise RuntimeError(f'RLE overlap QA failed: {name}')
            decoded.append(mask);mask_areas.append(int(mask.sum()));rle_checks+=1
        decoded_by_image[name]=len(decoded)
    if not set(empty)<=test_set:raise RuntimeError('Empty-image manifest contains unknown test ID')

    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(output)+'.tmp')
    frame.to_csv(tmp,index=False,lineterminator='\n');tmp.replace(output)
    if download.exists():raise FileExistsError(f'Refusing to overwrite {download}')
    shutil.copyfile(output,download)
    output_hash=sha(output)
    if output_hash!=sha(download):raise RuntimeError('Repository and Downloads CSV hashes differ')
    rebuilt=pd.read_csv(output)
    if list(rebuilt.columns)!=['filament_id','segmentation_rle'] or len(rebuilt)!=len(frame):raise RuntimeError('CSV rebuild/schema check failed')
    if not rebuilt.filament_id.is_unique:raise RuntimeError('CSV rebuild found duplicate filament IDs')
    audit={'status':'complete_test_csv_full_qa_passed','gate_path':str(GATE_PATH.relative_to(ROOT)),'gate_source':'repaired nested validation_gate.json only','validation_gate_passed':bool(gate['promote_to_test']),'validation_exposure':'Legacy R25 validation is exposed development, not OOF.','candidate_checkpoint_sha256':expected_ckpt_hash,'candidate_tensor_fingerprint':loaded_tensor_hash,'test_images':180,'validation_images':141,'test_validation_id_overlap':0,'test_validation_content_hash_overlap':0,'image_hashes_verified_against_manifest':180,'test_images_accounted_for':len(decoded_by_image),'empty_images':empty,'nonempty_test_images':len(decoded_by_image)-len(empty),'rows':len(frame),'unique_filament_ids':bool(frame.filament_id.is_unique),'rle_masks_checked':rle_checks,'all_rles_decode_to_2048x2048_nonempty':True,'per_image_masks_nonoverlapping':True,'schema':list(frame.columns),'csv_rebuild_check':'passed','csv_sha256':output_hash,'submission':str(output.relative_to(ROOT)),'download':str(download),'uploaded':False,'leaderboard_result':None,'test_label_scoring':'not performed'}
    write_json_atomic(OUT/'submission_qa.json',audit)
    write_json_atomic(OUT/'delivery.json',audit)
    write_json_atomic(STATE_PATH,{'phase':'complete','pid':None,'csv':str(output.relative_to(ROOT)),'download':str(download),'sha256':output_hash,'rows':len(frame),'test_images':180,'rle_checks':rle_checks,'candidate_checkpoint_sha256':expected_ckpt_hash,'validation_gate_path':str(GATE_PATH.relative_to(ROOT)),'validation_gate_passed':True,'qa':'passed','uploaded':False,'completed_local':time.strftime('%Y-%m-%dT%H:%M:%S')})
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
