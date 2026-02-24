# MedGemma Clinical Fine-Tuning

Fine-tunes [google/medgemma-4b-pt](https://huggingface.co/google/medgemma-4b-pt) on clinical trajectory data using an ODE-RNN + soft-prompt + LoRA architecture.

## Architecture

```
[values, times] → ClinicalODERNN → trajectory_embed (64-dim)
                                          |
                         Projection Layer (64 → 2560)
                                          |
                          [trajectory_prefix_tokens] + [text_tokens]
                                          |
                  MedGemma-4B (LoRA r=8, 4-bit quantized) → hidden_states
                                          |
                   Classification Head  OR  Causal LM Head
```

The ODE-RNN encodes irregular clinical time-series into trajectory embeddings, which are projected as soft-prompt tokens prepended to MedGemma's input. This lets the model reason over physiological dynamics that can't be expressed in plain text — irregular sampling, non-linear decay, multi-variate signals.

## Supported Datasets

| Dataset | Features | Use case |
|---|---|---|
| DKA (MIMIC) | Anion gap trajectory | ICU escalation prediction |
| SRTR | 4 features (Creatinine, Albumin, GFR, Dialysis) | Transplant waitlist mortality |

## Requirements

```bash
pip install -r requirements.txt
huggingface-cli login   # MedGemma is gated — accept license at hf.co/google/medgemma-4b-pt
```

Requires a CUDA GPU. Tested on A100 40GB. MedGemma-4B with 4-bit quantization uses ~3 GB VRAM.

## Usage

### Step 1 — Validate ODE-RNN embeddings (no LLM cost)

```bash
python medgemma_trainer.py --trajectory_only --dataset dka \
    --data_path /path/to/dka_training_data.csv
```

Target: val AUC > 0.68 before proceeding to full training.

### Step 2 — Fine-tune MedGemma (classification)

```bash
python medgemma_trainer.py --mode classification --dataset dka \
    --data_path /path/to/dka_training_data.csv \
    --ode_checkpoint ./checkpoints_medgemma_dka/medgemma_trajectory_best.pt
```

### Step 3 — Fine-tune MedGemma (generative)

```bash
python medgemma_trainer.py --mode generative --dataset dka \
    --data_path /path/to/dka_training_data.csv \
    --ode_checkpoint ./checkpoints_medgemma_dka/medgemma_trajectory_best.pt
```

### SRTR (transplant waitlist)

```bash
# Trajectory-only
python medgemma_trainer.py --trajectory_only --dataset srtr --organ kidney \
    --data_dir /path/to/srtr_data/

# Full training
python medgemma_trainer.py --mode classification --dataset srtr --organ kidney \
    --data_dir /path/to/srtr_data/ --max_samples 10000
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `classification` | `classification` or `generative` |
| `--dataset` | `dka` | `dka` or `srtr` |
| `--organ` | `kidney` | `kidney` or `liver` (SRTR only) |
| `--data_path` | `dka_training_data.csv` | Path to DKA CSV |
| `--data_dir` | — | Path to SRTR data directory |
| `--max_samples` | `10000` | Max patients (0 = all) |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `16` | Batch size |
| `--trajectory_only` | — | Train ODE-RNN only, skip LLM |
| `--trajectory_epochs` | `100` | Epochs for trajectory-only training |
| `--ode_checkpoint` | — | Pretrained ODE-RNN `.pt` (auto-freezes ODE-RNN) |
| `--freeze_ode` | — | Freeze ODE-RNN without loading checkpoint |
| `--time_scale` | `72.0` (DKA) | ODE-RNN time normalization in hours |
| `--patience` | `10` | Early stopping patience |
| `--log_file` | `auto` | Log path (`auto` = timestamped, `none` = disable) |

## DKA Data Format

CSV with two columns:

```
clinical_trajectory,label_escalation
"24 (0h) -> 20 (2h) -> 16 (4h) -> 12 (6h)",0
"24 (0h) -> 22 (24h) -> 20 (48h) -> 18 (72h)",1
```

- `label_escalation`: `0` = resolved / no escalation, `1` = escalation / poor response

## Results (DKA, A100)

| Mode | Val AUC | Test AUC | Test Acc |
|---|---|---|---|
| Trajectory-only (ODE-RNN) | 0.684 | — | — |
| Classification (+ MedGemma LoRA) | 0.690 | **0.761** | 72.0% |

## Files

| File | Description |
|---|---|
| `medgemma_trainer.py` | Main training script (self-contained) |
| `dte_encoder.py` | `ClinicalODERNN` — irregular time-series encoder via ODE solver |
| `dataset-format.py` | `MIMICtrajectoryDataset` for DKA/MIMIC data |
| `srtr_dataset.py` | `SRTRTrajectoryDataset` for transplant waitlist data |
