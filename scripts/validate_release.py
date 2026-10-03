"""CPU-only source checks. Does not import models or execute launchers."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT=Path(__file__).resolve().parents[1]
OMIT={'.git','__pycache__','.pytest_cache','.ruff_cache','.venv','workspace','outputs','runs'}
PATTERNS={
    'private key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'GitHub token':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'API token':r'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}',
    'AWS access key':r'AKIA[0-9A-Z]{16}',
    'Hugging Face token':r'hf_[A-Za-z0-9]{30,}',
    'private deployment path':r'/(?:cpfs[0-9]*|pfs/pfs-[A-Za-z0-9_-]+|Users)/[A-Za-z0-9_.-]+/',
}

def main():
    counts=dict(files=0,python=0,json=0,toml=0,shell=0,bytes=0)
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(x in OMIT for x in rel.parts) or p.name=='.env':continue
        if not p.is_file():continue
        if p.is_symlink():raise ValueError(f'Unexpected symlink: {rel}')
        if any(x in str(rel).lower() for x in ['robocasa','ablation','semantic_20k','native_baseline']):
            raise ValueError(f'Out-of-scope file: {rel}')
        text=p.read_text()
        if p.stat().st_size>5*1024*1024:raise ValueError(f'Unexpected large file: {rel}')
        for name,pattern in PATTERNS.items():
            if re.search(pattern,text):raise ValueError(f'{name} pattern in {rel}; value not printed')
        counts['files']+=1;counts['bytes']+=p.stat().st_size
        if p.suffix=='.py':ast.parse(text,filename=str(rel));counts['python']+=1
        if p.suffix=='.json':json.loads(text);counts['json']+=1
        if p.suffix=='.toml':tomllib.loads(text);counts['toml']+=1
        if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True);counts['shell']+=1
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for rel,expected in manifest.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source manifest mismatch: {rel}')
    print(json.dumps(dict(passed=True,counts=counts,source_hashes_verified=len(manifest)),indent=2))

if __name__=='__main__':main()
