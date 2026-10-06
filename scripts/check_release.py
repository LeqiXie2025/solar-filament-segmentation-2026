"""Verify/copy exact 0.38 prerequisites; standard library only, no training/upload."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL=Path("experiments/route_recovery_20261004/r25_proposal/repair_01")
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--mode",choices=["export","train","validate"],default="export")
 p.add_argument("--asset-root",type=Path,help="Read-only audit of assets in an existing project; does not change runtime paths")
 p.add_argument("--copy-from",type=Path,help="Copy missing verified assets into this clone; never overwrite")
 a=p.parse_args()
 if a.asset_root and a.copy_from:p.error("choose --asset-root or --copy-from")
 asset_root=(a.asset_root or ROOT).resolve(); source=a.copy_from.resolve() if a.copy_from else None
 pre=json.loads((ROOT/REL/"preregistration.json").read_text(encoding="utf-8"))
 gate=json.loads((ROOT/REL/"validation_gate.json").read_text(encoding="utf-8"))
 requirements={}
 for k,v in pre["sources"]["path_by_key"].items():requirements[v]=(pre["sources"]["sha256_by_key"][k],k)
 for arm in (["candidate_coco"] if a.mode=="export" else ["control_cv2","candidate_coco"] if a.mode=="validate" else []):
  rel=(REL/(arm+"_epoch6.pth")).as_posix()
  hashes={k.replace(chr(92),"/"):v for k,v in gate.get("source_hashes",{}).items()}
  requirements[rel]=(hashes.get(rel) if a.mode=="export" else None,arm+" checkpoint")
 names=(pre["split"]["train_files"]+pre["split"]["validation_files"] if a.mode=="validate" else pre["split"]["train_files"] if a.mode=="train" else pre["split"]["validation_files"])
 for name in names:requirements["data/raw/train_images/"+name]=(None,"official train image")
 if a.mode=="export":
  for im in pre["test_output"]["test_images"]["images"]:requirements["data/raw/test_images/"+im["file_name"]]=(im["sha256"],"official test image")
 bad=[];copied=0
 for rel,(expected,label) in requirements.items():
  # Repository sources are always checked in this checkout, not the external asset root.
  tracked=(ROOT/rel).is_file() and rel.endswith((".py",))
  target=(ROOT/rel if tracked else asset_root/rel).resolve()
  if not target.is_relative_to(ROOT if tracked else asset_root):raise ValueError("path outside permitted root")
  if not target.exists() and source:
   src=(source/rel).resolve()
   if not src.is_relative_to(source):raise ValueError("source path escaped root")
   if src.is_file() and (expected is None or digest(src)==expected):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);copied+=1
  if not target.is_file():bad.append({"path":rel,"issue":"MISSING","purpose":label})
  elif expected and digest(target)!=expected:bad.append({"path":rel,"issue":"SHA256_MISMATCH","purpose":label})
 # Included release code hashes remain frozen even after unrelated-file cleanup.
 for item in json.loads((ROOT/REL/"release_manifest.json").read_text(encoding="utf-8")).get("release_code_sha256",[]):
  f=ROOT/item["path"]
  if not f.is_file() or digest(f)!=item["sha256"]:bad.append({"path":item["path"],"issue":"RELEASE_CODE_CHANGED"})
 if a.mode=="export" and not gate.get("promote_to_test"):bad.append({"path":"validation_gate.json","issue":"GATE_NOT_PASSED"})
 result={"mode":a.mode,"checked_files":len(requirements),"copied_files":copied,"missing_or_invalid":bad,"status":"PASS" if not bad else "BLOCKED"}
 print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if not bad else 1
if __name__=="__main__":sys.exit(main())
