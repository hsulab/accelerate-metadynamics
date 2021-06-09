#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ===== VASP =====
VASP_WORKDIR = './vasp-worker'
VASP_COMMAND = os.environ['VASP_COMMAND']

VASP_PARAMS = dict(
    # INCAR
    system='Pt13-CO', nwrite=2, istart=0,
    lcharg = False, lwave= False,
    npar = 4,
    xc='PBE', encut=400, prec='Normal', ediff=1E-5, nelm=180, nelmin=6, 
    ispin=2, lorbit = 10,
    ismear=1, sigma = 0.2,
    algo = 'Fast', lreal = 'Auto', isym = 0,
    nsw=0,
    # KPOINTS
    kpts=(1,1,1), gamma=True,
)

# ===== QUIP and GAP =====
QUIP_COMMAND = os.environ['QUIP_COMMAND']
GAPFIT_EXEC = os.environ['GAPFIT_EXEC']

GAPFIT_XMLNAME = 'GAP.xml'
ENERGY_SIGMA = 0.008
FORCES_SIGMA = 0.04
VIRIAL_SIGMA = 0.04
HESSIAN_SIGMA = 0.0

GAPFIT_COMMAND = (
    GAPFIT_EXEC +
    " energy_parameter_name=free_energy" + 
    " force_parameter_name=forces" + 
    " virial_parameter_name=virial" + 
    " do_copy_at_file=F" + 
    " sparse_separate_file=F" + 
    " gp_file=%s" %GAPFIT_XMLNAME + 
    " at_file=./train.xyz" + 
    " default_sigma={%f %f %f %f}" %(ENERGY_SIGMA, FORCES_SIGMA, VIRIAL_SIGMA, HESSIAN_SIGMA) + 
    " e0={C:0.0:O:0.0:Pt:0.0}" + 
    " gap={" + 
    " distance_2b cutoff=5.000000 Z1=6 Z2=8 covariance_type=ard_se delta=0.200000 theta_uniform=0.500000 sparse_method=uniform n_sparse=50 add_species=F : " +
    " distance_2b cutoff=5.000000 Z1=6 Z2=78 covariance_type=ard_se delta=0.200000 theta_uniform=0.500000 sparse_method=uniform n_sparse=50 add_species=F : " +
    " distance_2b cutoff=5.000000 Z1=8 Z2=78 covariance_type=ard_se delta=0.200000 theta_uniform=0.500000 sparse_method=uniform n_sparse=50 add_species=F : " +
    " distance_2b cutoff=5.000000 Z1=78 Z2=78 covariance_type=ard_se delta=0.200000 theta_uniform=0.500000 sparse_method=uniform n_sparse=50 add_species=F : " +
    " {soap       cutoff=5.000000 covariance_type=dot_product delta=0.20000 sparse_method=cur_points n_sparse=1200 zeta=2 l_max=6 n_max=12 atom_sigma=0.5} " +
    " }"
)

