#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase import units

from ase.io import read, write
from ase.calculators.vasp import Vasp2

from ase.constraints import FixAtoms

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from AccMetaD.nosehoover import NoseHoover
from AccMetaD.md_utils import force_temperature
from AccMetaD.MetaAMLP import MetaAMLP
from AccMetaD.xquip import XQuip
from AccMetaD.xdftb import XDFTB

if __name__ == '__main__':
    import inputs # all parameters

    # ===== system =====
    atoms = read('./Pt13-CO.xyz')

    cons = FixAtoms(indices=[2])
    atoms.set_constraint(cons)

    # - basic
    dftb_shared_params = inputs.generate_calculator(atoms)
    dftb_shared_params.update(command = inputs.DFTB_COMMAND)
    dftb_shared_params.update(num_threads = 20) # OMP_THREADS
    dftb_shared_params.update(slako_dir = inputs.DFTB_SLAKO)
    dftb_shared_params.update(kpts = (1,1,1))

    dftb_shared_params.update(binding_energy = True)
    dftb_shared_params.update(
        atomic_energies = dict(
            energy = {'C': -38.2611, 'O': -84.4551, 'Pt': -69.2691}, 
            free_energy = {'C': -38.4153, 'O': -84.6094, 'Pt': -69.2697}
        )
    )

    dftb_params = dftb_shared_params.copy()
    dftb_params.update(directory = './dftb-worker')

    # - correction
    correction_params = dict(
        correction_command = inputs.QUIP_COMMAND, 
        param_filename = './GAP.xml',
        calc_args = 'local_gap_variance', 
    )

    xdftb_params = dftb_shared_params.copy()
    xdftb_params.update(
        dict(
            directory = './xdftb-worker', 
            correction_params = correction_params
        )
    )

    # - reference
    vasp_params = inputs.VASP_PARAMS.copy()
    vasp_params.update(
        dict(
            command=inputs.VASP_COMMAND,
            directory=inputs.VASP_WORKDIR,
        )
    )

    # - calculators
    calculators = {
        'basic': XDFTB,
        'corrected': XDFTB,
        'reference': Vasp2,
    }

    calcParams ={
        'basic': dftb_params,
        'corrected': xdftb_params,
        'reference': vasp_params,
    }

    # ===== molecular dynamics =====
    timestep = 2.0 * units.fs
    temperature = 300 # in Kelvin

    MaxwellBoltzmannDistribution(atoms, temperature*units.kB)
    force_temperature(atoms, temperature)

    trainer = MetaAMLP(
        atoms = atoms,
        constraints = cons, 
        calculators = calculators,
        calcParams = calcParams,
        gapfitCommand = inputs.GAPFIT_COMMAND,
        # md stages 
        mdEngines = [NoseHoover, NoseHoover], 
        mdParams = [
            dict(
                nsteps = 0,
                plumedInput = 'plumed-1.dat',
                nvt_q = 334., temperature = temperature*units.kB
            ),
            dict(
                nsteps = 20000,
                init_atoms = atoms.copy(), 
                plumedInput = 'plumed-2.dat',
                nvt_q = 334., temperature = temperature*units.kB
            )
        ], 
        timestep = timestep, # fixed timestep
        # sample details
        maxSteps = 60000,
        minSamples = 1,
        sampleInterval = 2,
        tolerances = {
            'energy': 1000, 
            'forces': 0.04,
            'force_percent': 0.20,
            'force_threhold': 0.02
        },
        retrain_criteria = {
            'min_interval': 10, 'max_interval': 1000, 'new_samples': 5
        }, 
        restart = False, 
    )

    trainer.run()
