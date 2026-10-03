"""Verify the published weight manifest with streaming SHA256; no model loading."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args();root=args.directory.resolve()
    entries=json.loads((root/'SHA256SUMS.json').read_text())
    for item in entries:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Manifest path leaves the download directory')
        if not path.is_file() or path.stat().st_size!=item['size']:raise ValueError('Missing or truncated: '+item['path'])
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
        if h.hexdigest()!=item['sha256']:raise ValueError('Checksum mismatch: '+item['path'])
        print('OK',item['path'],flush=True)
    print('Verified',len(entries),'files')

if __name__=='__main__':main()
