#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# ===== DFTB =====
DFTB_COMMAND = os.environ['DFTB_COMMAND']
DFTB_SLAKO = os.environ['DFTB_SLAKO']

MAX_ANG_MOM = {'C': 'p', 'O': 'p', 'Pt': 'd'}

basic_input_parameters = dict(
    # Hamiltonian
    Hamiltonian_SCC = 'Yes',
    Hamiltonian_SCCTolerance = '1.0E-5',
    Hamiltonian_MaxSCCIterations = '180',
    Hamiltonian_Mixer_ = 'Broyden',
    Hamiltonian_Mixer_MixingParameter = '0.2',
    # Filling
    Hamiltonian_Filling_ = 'MethfesselPaxton',
    Hamiltonian_Filling_Order = '1',
    Hamiltonian_Filling_Temperature = '0.00735',
    # force
    Hamiltonian_Differentiation_ = 'FiniteDiff',
    Hamiltonian_Differentiation_Delta = '0.001',
    Hamiltonian_ForceEvaluation = 'traditional',
    # Options
    Options_ = ' ',
    Options_WriteResultsTag = 'Yes',
    # Analysis
    Analysis_ = ' ',
    Analysis_CalculateForces = 'Yes',
    # ParseOptions
    ParserOptions_ = ' ',
    ParserOptions_ParserVersion = '7',
    # Parallel
    Parallel_ = ' ',
    #Parallel_Groups = '2', # MPI 
    Parallel_UseOmpThreads = 'Yes',
)

def generate_calculator(
        atoms, 
        run_nsteps = 0, 
        pure_elec = False, 
        cell_opt = False
    ):
    # check elements
    symbols = atoms.get_chemical_symbols()
    elements = list(set(symbols))
    nelements = len(elements)

    # ===== Initialize Parameters
    input_parameters = basic_input_parameters.copy()

    # ===== Element Related
    # angular
    input_parameters.update({'Hamiltonian_MaxAngularMomentum_': ''})
    for elm in elements:
        input_parameters.update(
            {'Hamiltonian_MaxAngularMomentum_%s'%elm: '\"%s\"' %MAX_ANG_MOM[elm]})

        # no rep
        if pure_elec:
            input_parameters.update({'Hamiltonian_PolynomialRepulsive_': ''})

            two_body = []
            for e1 in range(nelements):
                for e2 in range(nelements):
                    two_body.append((e1,e2))

            for pair in two_body:
                elm1, elm2 = elements[pair[0]], elements[pair[1]]
                input_parameters.update({'Hamiltonian_PolynomialRepulsive_%s-%s' \
                    %(elm1,elm2): 'Yes'})

        #print(input_parameters)
    # ===== KPOINTS
    #cur_kpts = (6,6,6)
    #if len(atoms) < 6: # Pt bulk, O2, CO, CO2
    #    cur_kpts = (6,6,6)
    #elif len(atoms) < 30: # surf, adsorption, reaction
    #    cur_kpts = (4,4,1)
    #else:
    #    raise ValueError('Wrong KPOINTS!!!')

    # ===== OPTIMIZATION / Molecular Dynamics
    # Driver
    input_parameters.update(Driver_='ConjugateGradient')
    input_parameters.update(Driver_MaxForceComponent='0.0015')
    input_parameters.update(Driver_MaxAtomStep= '0.2')
    input_parameters.update(Driver_MaxSteps=run_nsteps)
    input_parameters.update(Driver_AppendGeometries='Yes')

    if cell_opt:
        input_parameters.update(Driver_LatticeOpt = 'Yes')
        input_parameters.update(Driver_FixAngles = 'Yes')
        input_parameters.update(Driver_FixLengths = '{ No No No }')
        input_parameters.update(Driver_MaxLatticeStep = '0.1')

    return input_parameters


# ===== VASP =====
VASP_WORKDIR = './vasp-worker'
VASP_COMMAND = os.environ['VASP_COMMAND']

VASP_PARAMS = dict(
    # INCAR
    system='CO-Pt111', nwrite=2, istart=0,
    lcharg = False, lwave= False,
    npar = 4,
    xc='PBE', encut=400, prec='Normal', ediff=1E-5, nelm=120, nelmin=6, 
    ispin=1, 
    ismear=1, sigma = 0.2,
    algo = 'Fast', lreal = 'Auto', isym = 0,
    nsw=0,
    # KPOINTS
    kpts=(4,4,1), gamma=True,
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

