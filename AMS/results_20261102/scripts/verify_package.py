#!/usr/bin/env python3
"""Verify the delivered supplement against its SHA256 manifest, without a GPU."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'package_manifest.json').read_text())
    errors = []
    for relative, recorded in manifest['files'].items():
        path = root / relative
        if not path.is_file():
            errors.append(f'MISSING: {relative}')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != recorded['bytes'] or digest.hexdigest() != recorded['sha256']:
            errors.append(f'CHANGED: {relative}')
    for error in errors:
        print(error)
    print(f"Checked {len(manifest['files'])} recorded files; {len(errors)} errors.")
    print('This checks delivery integrity, not statistical or algorithmic validity.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
