# mert_cft

This repository is a minimal handoff package centered on the successful MERT+CFT experiment:

- run id: `20260614_011400_COnP`
- training script: `train_conp_v6_0415.py`
- config: `config_mert_base.yaml`
- model selection target: `COnP F1`

It intentionally excludes unrelated experiments, alternative batch-size runs, historical checkpoints, and bulky output folders.

## Included

- `train_conp_v6_0415.py`: training entry
- `model.py`: MERT frontend + CFT model
- `dataset.py`: MIR-ST500 dataset loader with 3-source mixed training support
- `predict_to_json.py`: prediction export
- `evaluate_github.py`: evaluation script
- `config_mert_base.yaml`: exact config used by the target run
- `data/MIR-ST500_corrected.json`: label file used by this setup
- `splits_v11/`: split files used by this run
- `run_records/20260614_011400_COnP/`: key logs from the target experiment

## Exact Training Conditions

Evidence comes from `run_records/20260614_011400_COnP/logs/train_stdout.log`.

- Dataset input: waveform
- Sample rate: `24000`
- Segment frames: `512`
- Infer chunk frames: `512`
- Max samples per epoch: `12000`
- Train mix sources: `original + pitch_shift + pitch_shift_complement`
- Frontend: `MERT-v1-95M`
- Frontend freeze feature encoder: `true`
- Batch size: `8`
- Num workers: `4`
- Learning rate: `1e-4`
- Backbone learning rate: `1e-5`
- Epochs: `1300`
- Scheduler: `CosineAnnealingLR`
- AMP: enabled
- Best-model criterion: `COnP F1`

## Main Result Snapshot

From `run_records/20260614_011400_COnP/test_monitor.txt`:

- best test-monitor `COnP_f1`: `0.810985` at epoch `8`
- best test-monitor `COnPOff_f1`: `0.638390` at epoch `12`
- best test-monitor `COn_f1`: `0.837046` at epoch `8`

The validation-side log continued improving later, but the monitored holdout test peak for `COnP` in the retained record is epoch 8.

## Paths You Must Adjust

`config_mert_base.yaml` still points to the original local data/model paths. Before running on a new machine, update at least:

- `data.audio_dir`
- `data.label_path`
- `data.splits_dir`
- `data.cqt_cache_dir`
- `data.train_mix_sources[*]`
- `model.wav2vec_path`
- `training.run_dir`

## Run

Use the same environment style recorded in:

- `run_records/20260614_011400_COnP/reproduce_successful_run.md`

Training command:

```bash
python3 train_conp_v6_0415.py --config config_mert_base.yaml
```

## Notes

- This package is intentionally narrow: it is for this MERT+CFT experiment only.
- No other experiments were copied into this repository.
