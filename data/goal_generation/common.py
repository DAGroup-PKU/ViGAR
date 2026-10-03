"""Local artifact helpers shared by preparation, generation and cache export."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_config import CONFIG, CONFIG_SHA256, require_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path leaves its workspace')
    return path


def load_prepared(root):
    contract = json.loads((root / 'contract.json').read_text())
    require_config(contract)
    receipt = json.loads((root / 'PREPARED.json').read_text())
    if receipt['contract_sha256'] != sha(root / 'contract.json') or receipt['cases_sha256'] != sha(root / 'cases.json'):
        raise ValueError('Prepared inputs changed')
    return contract, json.loads((root / 'cases.json').read_text())
