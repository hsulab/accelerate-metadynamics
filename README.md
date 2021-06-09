# Accelerating Metadynamics-Based Free-Energy Calculations with Adaptive Machine Learning Potentials

This repository contains codes for paper [Xu, J.; Cao, X.-M.; Hu, P. JCTC, 2021.](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00261).

This package aims to accelerate metadynamics (MetaD) in heterogeneous reactions using adaptive machine learning potentials (AMLP) that maintains *ab initio* accuracy.

## Installation

Install the external codes, prepare the python environment and add `AccMetaD` to `PYTHONPATH`.

### External Codes

1. VASP 5.4.1
2. DFTB+ 20.1
3. QUIP with GAP

### Python Packages

1. ase 3.19.1
2. plumed 2.6.2

### Notes

1. Other DFT codes can be utilised as well, which can be accessed by the ase interface.
2. Units in dynamics modules have been changes in ase 3.21.0. Change timestep and temperature accordingly if using new version of ase.

## Usage

### Introduction
Each job contains at least five input files. 

`*.xyz` is the structure.

`plumed-*.dat` are inputfiles for plumed.

`inputs.py` contains DFT, DFTB and GAP calculation parameters.

`run.py` contains AMLP-MetaD settings.

`acc_meta.slurm` is the job script that sets environment variables.

### Examples

There are four examples attached.

CO on Pt13 cluster using GAP and DFTB-GAP.

CO on Pt(111) surface using GAP and DFTB-GAP.

