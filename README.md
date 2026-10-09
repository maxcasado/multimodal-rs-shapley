# shapley-fusion-cropharvest

Reproduction code for the paper Shapley Values for Modality Attribution in Robust Multimodal Remote Sensing Models

The paper measures how much each sensor contributes to multi-sensor crop / non-crop
classifiers on [CropHarvest](https://github.com/nasaharvest/cropharvest). The four sensors
are Sentinel-2 + NDVI (S2), Sentinel-1 (S1), weather and DEM. Contributions are measured with
exact **Shapley values** over the 2⁴ = 16 sensor coalitions. They are compared with the
**Perceptual Score**, decomposed with **Shapley interactions** and Harsanyi dividends, and
studied at two levels:
- **globally**, with the fold-level f1 as characteristic function;
- **per sample**, as local maps.

Two S2 degradations are studied: Gaussian noise and cloud gaps.

This repository starts from the preprocessed data. It trains every model, computes
every game and regenerates **every figure, table and number** of the paper. No trained
weights are distributed: the pipeline trains them. A full run takes about 11 GPU-hours.
The paper's per-sample Shapley values are distributed in
[`shapley_local.geojson`](#local-shapley-values-geojson) at the repository root.

```bash
pip install -r requirements.txt      # Python 3.10, CUDA 11.8 (see requirements.txt for CPU)
make data                            # dataset + Natural Earth, md5-checked (~70 MB)
make smoke                           # the whole pipeline on a 3,000-sample subset (~15 min)
make all                             # the paper: train, infer, report  (~11 h on 1 GPU)
```

Every figure, table and number then sits in `paper_outputs/`.

---

## What is reproduced

| Deliverable | File(s) in `paper_outputs/` | Report step |
|---|---|---|
| φ vs Gaussian noise σ on S2, CoM \| EmbraceNet | `figures/fig_noise_shapley.png` | A |
| Perceptual Score vs σ | `figures/fig_noise_ps.png` | A |
| φ = φ_grand + φ_small | `figures/fig_phi_decomposition.png` | A |
| φ vs share of S2 dates lost (cloud gaps) | `figures/fig_cloud_gap_shapley.png` | A |
| f1 of the full model vs share of S2 dates lost | `figures/fig_cloud_gap_f1.png` | A |
| Local Shapley maps (2° cells) | `figures/fig_shapley_local_map_{CoM,EmbraceNet}.png` | A |
| f1 of the 16 coalitions (main, noise, cloud gaps) | `tables/table_coalitions_f1{,_noise,_cloud_gap}.txt` (LaTeX) | B |
| Shapley vs Perceptual Score tables | `tables/tables_shapley_ps.txt` (LaTeX) | C |
| Global interactions under degradation | `figures/interactions_degradation.pdf` | D |
| Local interaction maps (S2 × weather; 6 pairs) | `figures/interactions_local_{s2_weather,pairs}.pdf` | E |
| Every number quoted in the text | `numbers/report_numbers.{md,json}`, `numbers/interactions_local.md` | F |
| Accuracy / f1 of the four models; local maps of all four | `tables/table_metrics_global.txt`, `figures/models_overview/` | G |
| Supplementary local analyses (robustness, model comparison, value functions, missing sensors, hypotheses, interaction decomposition) | `local_analysis/p0_checks … p5_interactions/`, `local_analysis/VALUE_FUNCTION.md` | H |
| Local Shapley values of every sample, CoM and EmbraceNet (GeoJSON) | `shapley_local.geojson`, at the repository root | I |

The CSV/JSON data behind every figure is in `paper_outputs/data/`.
`python -m shapfusion report --only A B` regenerates a subset.

## Local Shapley values (GeoJSON)

[`shapley_local.geojson`](shapley_local.geojson) holds the values behind the paper's local maps:
one point (lon/lat, WGS84) per CropHarvest sample, 69,800 in all, each explained by the model of
its test fold.

| Property | Meaning |
|---|---|
| `identifier`, `fold`, `y_true` | CropHarvest identifier, test fold, label (1 = crop) |
| `<model>_yhat` | class predicted with the four sensors, `<model>` ∈ {`CoM`, `EmbraceNet`} |
| `<model>_phi_<sensor>` | local φ of the sensor, `<sensor>` ∈ {`S2`, `S1`, `weather`, `DEM`}, 4 decimals |

The game is the local game defined below, so φ > 0 means the sensor supports the class the model
predicts, and the four φ of a model sum to P(ŷ_full | x) − 0.5. The file opens in QGIS or with
`geopandas.read_file("shapley_local.geojson")`.

The versioned file comes from the paper's runs. `make geojson` (step I of the report, also run by
`make figures`) rewrites it from the artefacts in `runs/`: after a retraining it holds the values
of the new run, not those of the paper (`git checkout shapley_local.geojson` restores them).
With `--smoke` it is written to `paper_outputs_smoke/` instead.

## Models

All four models use a TempCNN encoder for each time series (12 dates) and an MLP for DEM.
They are trained with **sensor dropout** (each sensor dropped with p = 0.5). A missing sensor
is never imputed: its branch does not run.

| Name | Fusion | Recipe |
|---|---|---|
| `CoM` | decision level: mean of the class logits of the available sensors | `configs/models/CoM.yaml` |
| `EmbraceNet` | feature level: docking layers + embracement (stochastic, also at test time) | `configs/models/EmbraceNet.yaml` |
| `DSensDp_noLN` | DSensD+ (two passes per step + mutual distillation), no LayerNorm on the logits | `configs/models/DSensDp_noLN.yaml` |
| `DSensDp` | DSensD+ as the reference code (LayerNorm on the 2 logits: the branches cannot train; kept for reference) | `configs/models/DSensDp.yaml` |

## The pipeline

Every parameter lives in [`configs/paper.yaml`](configs/paper.yaml): folds and seeds, noise
levels, cloud-gap levels, number of draws and permutations, and the models of each
experiment.

**Jobs.** The unit of work is a job = (model, experiment, level, fold), 220 in total:

| Experiment | Models | Levels | Jobs |
|---|---|---|---|
| `main` | all 4 | — | 20 |
| `noise` | CoM, EmbraceNet | σ ∈ {0, 0.1, 0.25, 0.5, 1, 2, 5} | 70 |
| `cloud_gap` | CoM, EmbraceNet | k ∈ {0, …, 12} of the 12 S2 dates | 130 |

**Stages.** Each job runs up to four stages, written to `runs/<model>/<experiment>[/<level>]/fold<k>/`:

| Stage | Output | What it does |
|---|---|---|
| `train` | `model.pt`, `metrics.json` | trains on the 4 other folds (5-fold CV, seed 10); the noise / gaps are applied to train AND test data |
| `vdict` | `vdict.pkl` | the 16-coalition game on the test fold: per-coalition predictions; for EmbraceNet, metrics averaged over R = 20 draws |
| `ps` | `ps.csv` | permutation Perceptual Score, 10 permutations (main and noise) |
| `local` | `local.npz` | per-sample v(S) of the 16 coalitions (main), for the local maps and analyses |

**Resuming.** Every stage is skipped when its output exists, so an interrupted run resumes
where it stopped.

**Report.** `python -m shapfusion report` then reads `runs/` (CPU only) and writes `paper_outputs/`.

Useful commands:

```bash
python -m shapfusion status                                # done / missing stages
python -m shapfusion run --experiments main --models CoM   # a subset (filters: --models --experiments --levels --folds)
python -m shapfusion run --stages vdict ps --force         # recompute stages from the saved weights
CUDA_VISIBLE_DEVICES=1 make cloud                          # second GPU, in parallel with `make noise`
```

### Definitions

- **Global game.** For a fold, v(S) = f1_weighted on the test fold of the model restricted
  to the sensors in S. v(∅) = 0.5127 is the expected f1_weighted of a uniform coin-flip
  classifier, 2p²/(2p+1) + 2(1−p)²/(3−2p), with p = 0.65744 the positive share of the data.
  It is applied when the games are read (`shapfusion/shapley/characteristic.py`).
- **Local game.** For a sample x, v_x(S) = P(ŷ_full(x) | x_S) and v_x(∅) = 0.5. The game is
  oriented on the class the full model predicts, so φ > 0 means the sensor supports the
  model's decision.
- **Interaction index.** Order-2 Shapley interaction (Grabisch–Roubens) with weights
  1/3, 1/6, 1/6, 1/3. I < 0 means substitution / redundancy, I > 0 complementarity.
- **Cross-fold statistics.** Every reported value is mean ± std (ddof = 1) over the 5 folds.
  Values with |mean/std| < 2 are flagged *unresolved* (hollow markers, † in tables, n.r. in text).


## Reproducibility notes

- **Same pipeline as the paper.** The code is that of the paper's runs, pruned to the
  pieces used. Running the report on the paper's own trained artefacts reproduces every
  CSV, table and number of the paper bit for bit. Running inference on the paper's
  checkpoints reproduces its v-dicts and Perceptual Scores exactly.
- **EmbraceNet local maps.** The local game of EmbraceNet averages the same R = 20 draws as
  its coalition values, so its local φ map and its local interaction maps come from one game.
- **Retraining does not give bit-identical weights.** The paper's runs were not seeded, and
  cuDNN / TF32 kernels are not deterministic. This repository seeds every training
  (`train_seed + fold`) and every stochastic draw, so a rerun is close to repeatable.
  Expect differences from the paper within its fold-to-fold standard deviations; the
  conclusions do not depend on them.
- **Early stopping uses the test fold.** As in the paper's runs, the validation set used for
  early stopping and for picking the best epoch is the test fold. It is therefore mildly
  optimistic for the absolute f1. It does not affect the comparisons between sensors, which
  all use the same model.
- **Dataset-wide quantities.** The z-score statistics (shipped with the data) and the p of
  v(∅) are computed over the whole dataset.
- **EmbraceNet is stochastic at test time.** Coalition values and local games average
  R = 20 draws. The Perceptual Score and `metrics_global` use one draw, as in the paper.

## Repository layout

```
configs/paper.yaml          every parameter (+ the --smoke overlay)
configs/models/*.yaml       the four model recipes
shapfusion/
  cli.py, job.py, train.py  command line, the job runner, Lightning training
  data/                     dataset, 5-fold split, S2 perturbations (noise, cloud gaps)
  models/                   encoders, fusions, DSensD+, model builder
  shapley/                  games, Shapley values / interactions, v(∅), Perceptual Score
  analysis/                 global games, house plot style, map helpers
  report/                   deliverables A–G, I
  local_analysis/           deliverable H (P0 checks, P1–P5)
tests/                      unit tests (no data needed): pytest -q
shapley_local.geojson       the paper's local Shapley values, one point per sample (deliverable I)
```

## Data

`make data` downloads the following, each checked against its md5:
- the preprocessed CropHarvest binary dataset (69,800 samples, 62 MB) and its statistics,
  from the DSensD+ authors' share (`https://cloud.dfki.de/owncloud/index.php/s/Xkd78A4BFZn9mRc`);
- Natural Earth 1:110m v5.1.1 (land, countries).

If the DFKI share moves, place `cropharvest_binary.nc` and `stats/stats_cropharvest_binary.nc`
in `data/` by hand (md5 in `shapfusion/download.py`). Third-party code and data are credited
in [NOTICE.md](NOTICE.md).

## Citation

See [CITATION.cff](CITATION.cff).
