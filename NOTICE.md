# Notice: third-party code and data

## Code

The model code in `shapfusion/models/` and `shapfusion/data/dataset.py` is adapted from
[fmenat/DSensDp](https://github.com/fmenat/DSensDp) (F. Mena et al., *Multi-sensor model for
Earth observation robust to missing data via sensor dropout and mutual distillation*,
IEEE Access 13, 2025). It has been pruned to the encoders and fusions used in this paper
(TempCNN, MLP, mean fusion, EmbraceNet, DSensD+); module and parameter names are unchanged.

- TempCNN: Pelletier et al., *Remote Sensing* 11(5), 2019.
- EmbraceNet: Choi & Lee, *Information Fusion* 51, 2019.
- Perceptual Score: Gat et al., NeurIPS 2021.

## Data (downloaded by `python -m shapfusion download`, not redistributed here)

- **CropHarvest** (Tseng et al., NeurIPS 2021 Datasets and Benchmarks,
  [nasaharvest/cropharvest](https://github.com/nasaharvest/cropharvest)), binary crop / non-crop
  task, in the preprocessed form distributed by the DSensD+ authors
  (`cropharvest_binary.nc` and its per-band statistics).
- **Natural Earth** 1:110m v5.1.1 (public domain,
  [nvkelso/natural-earth-vector](https://github.com/nvkelso/natural-earth-vector)): land
  polygons for the map backgrounds, countries for the continent of each sample.
