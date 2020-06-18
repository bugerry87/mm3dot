# Multi Model 3D Object Tracker

A 3D object tracker for pre-computed detections, currently based on Kalman Filter.

## Installation

**We strongly recommend to use [conda](https://docs.anaconda.com/anaconda/install/) environments since the dataset APIs require incompatible python versions!**

## Usage Examples

### Fake data

If you don't have any pre-computed detection results run an example for on-the-fly generated fake data.

```bash
python run.py fake
```

### Argoverse (ArgoAI)

For the Argoverse dataset you may download the detection results of [coming soon]().
Use wildcards to define a list of input files.
Replace `$DATAROOT` with where ever you used to store all the data.

```bash
python run.py argoverse \
	--inputfile '$DATAROOT/results/agroverse_detections_2020/validation/*/*/*.json' \
	--groundtruth '$DATAROOT/argoverse-tracking/val/*/per_sweep_annotations_amodal/*.json'
```
