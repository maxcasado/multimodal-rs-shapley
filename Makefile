# Reproduction of the paper, stage by stage (every stage resumes where it stopped).
#   make data     download the dataset + Natural Earth (md5-checked)
#   make smoke    the whole pipeline on a 3,000-sample subset (checks that everything runs)
#   make main     the 4 models x 5 folds on clean data             (GPU)
#   make noise    Gaussian-noise sweep on S2, retrained per sigma  (GPU)
#   make cloud    cloud-gap sweep on S2, retrained per level       (GPU)
#   make figures  every figure / table / number -> paper_outputs/  (CPU)
#   make geojson  only the local Shapley values -> shapley_local.geojson (also done by figures)
#   make all      data + main + noise + cloud + figures
#   make test     unit tests (no data needed)
# Two GPUs: run e.g. `CUDA_VISIBLE_DEVICES=0 make noise` and `CUDA_VISIBLE_DEVICES=1 make cloud`
# in two shells after `make main`.
PYTHON ?= python3
SF = $(PYTHON) -m shapfusion

.PHONY: data smoke main noise cloud train figures geojson all test status

data:
	$(SF) download

smoke: data
	$(SF) run --smoke
	$(SF) report --smoke

main:
	$(SF) run --experiments main

noise:
	$(SF) run --experiments noise

cloud:
	$(SF) run --experiments cloud_gap

train: main noise cloud

figures:
	$(SF) report

geojson:
	$(SF) report --only I

all: data train figures

status:
	$(SF) status

test:
	$(PYTHON) -m pytest -q
