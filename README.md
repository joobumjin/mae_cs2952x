# MAE : CS2952X

This repository is my implementation of FAIR's Masked AutoEncoder model!

Many of the util files are diretly imported from the original MAE repository (found [here](https://github.com/facebookresearch/mae))

## Setup

First, the conda environment can be created by calling
```bash
conda env create -f environment.yml
```

## Running Code

### Pretraining

All of the pretraining routine is contained within `simple_pretrain.py` and can be run by either executing the file directly from terminal (make sure to include relevant arguments), or by submitting the slurm request `run_simple.sh`.

Many of the pre-set filepaths will likely not work, so it is important to set those arguments before calling the file.

### Linear Probing

All of the linear probing routine is contained within `simple_linprobe.py` and can similarly be run by either executing the file directly from terminal or by submitting the slurm request `run_probe.sh`

This code will automatically generate a W&B run, so make sure to disable that functionality if it is not desirable.

To run the linear probe on FAIR's pretrained weights, you can submit the slurm request `run_probe_fbweights.sh`.

There is additional Optuna code that has been commented out to enable automated hyperparameter search. 
However, the computation time for a single linear probe model was surprisingly long, and so the code was not used.

### Reconstruction Visualization

The code for reconstruction vis is within `reconstruction_vis.py` and can be run by either executing the file directly from terminal or by submitting `run_reconvis.sh`. 

This code has hard-coded paths to saved weights, so make sure to update them to relevant filepaths before running.