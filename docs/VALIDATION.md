# Release validation

Public-source validation completed on 2026-10-03:

- **44 CPU tests passed, none skipped**: stage-selection boundaries, image processing, ROI projection and 4:1 gradients, RGB encoding, Strict Sync freshness, resumable seed queues, release binding, checksum-path safety and standalone data-tool imports.
- Python syntax, JSON/TOML parsing, shell syntax and repository-relative content hashes were checked.
- The fixed normalizer and 5000 paired seed/instruction entries retain their verified checksums.
- Internal deployment paths, personal account defaults, private host addresses, source-machine manifests and prior private Git history are excluded. Common credential patterns are checked before publication.

Run the CPU checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-project --no-config --python 3.12 \
  --with-requirements requirements-test.txt python -m pytest -q
python3.12 scripts/validate_release.py
```

Tests were run on macOS/Python 3.12 with PyTorch 2.14.1. These checks establish source and selected CPU contracts; they do not establish compatibility with every CUDA driver or reproduce a full GPU training/evaluation run.

Weights were staged without changing tensor payloads. Source checksums were verified, DCP storage-origin metadata was sanitized without changing tensor/storage mappings, and the policy's completion manifest was rebuilt and validated. The model repository supplies a streaming-verifiable SHA256 manifest.

This upload does not claim a new 5000-episode score for the selected policy 50k + ROI-I2I 70k combination. Historical scores from a different planner pairing must not be relabeled as results of this release.
