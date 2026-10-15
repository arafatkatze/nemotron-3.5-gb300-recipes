#!/usr/bin/env python3
"""Download the exact published model revision; no weights are distributed here."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Download approximately75GB of pinned NVIDIA model weights.')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'model')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4',
        revision='2e5cc3953bb1ea66b2fb7c831fee3aa317427e69',local_dir=str(args.output))
    manifest=json.loads((Path(__file__).resolve().parents[1]/'evidence/model-manifest.json').read_text())
    for row in manifest['files']:
        f=args.output/row['file']
        if f.stat().st_size!=row['bytes']:raise RuntimeError(f'Size mismatch: {row["file"]}')
        digest=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=row['sha256']:raise RuntimeError(f'SHA256 mismatch: {row["file"]}')
    print(f'Verified {len(manifest["files"])} pinned files.')
    print(args.output)

if __name__=='__main__':main()
