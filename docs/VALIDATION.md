# Release validation

Public-source validation completed on 2026-10-03:

- **49 CPU tests passed, none skipped**: stage-selection boundaries, image processing, ROI projection and 4:1 gradients, RGB encoding, Strict Sync freshness, resumable seed queues, release binding, checksum-path safety and standalone data-tool imports, direct final-task annotation, complete generated-goal export and configuration binding.
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

## Data preparation refactor

On 2026-10-03, the direct builder was compared with the previous two-step annotation builder and transform using a synthetic 50-task × 50-episode fixture, including nine single-bread episodes. All 2,500 episode annotations and 3,691 stage semantics matched after excluding version/provenance labels. The 50 task modes, retained stage texts and predicate descriptions also matched the prior final configuration. The comparison does not establish replay equivalence on real simulator trajectories.

CPU integration tests materialize the direct annotated dataset and export/verify a complete synthetic goal cache. They check 19 multi-stage / 31 final-goal tasks, 950 boundary records, 74 task/stage definitions, variable bread stages, official RGB channel order, selection provenance, rejected selections and path containment. Model inference, VLM calls, GPU replay and training were not run for this refactor. The pretrained weights, fixed normalizer and paired evaluation instructions are unchanged.
