import hashlib
import json
import os
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[2]/'training_runtime'
ROOT=Path(os.environ['GOALWAM_I2I_ASSET_ROOT'])
WORK=Path(os.environ['GOALWAM_I2I_WORK'])
CHECKPOINT=Path(os.environ['GOALWAM_I2I_NORMAL30K'])
DATASET=Path(os.environ['EPISODE_IMAGE_EDIT_DATASET_PATH'])
PREVIOUS=Path(os.environ['GOALWAM_I2I_METADATA_ROOT'])
OUTPUT=Path(os.environ['GOALWAM_I2I_OUTPUT'])
MODES=('ee96_mix19',)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def append_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value)+'\n')
