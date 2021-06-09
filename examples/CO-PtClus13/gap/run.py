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

if __name__ == '__main__':
    import inputs # all parameters

    # ===== system =====
    atoms = read('./Pt13-CO.xyz')

    cons = FixAtoms(indices=[2])
    atoms.set_constraint(cons)

    # - correction
    xquip_params = dict(
        command = inputs.QUIP_COMMAND, 
        param_filename = './GAP.xml',
        calc_args = 'local_gap_variance', 
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
        'basic': None,
        'corrected': XQuip,
        'reference': Vasp2,
    }

    calcParams ={
        'basic': None,
        'corrected': xquip_params,
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
            'energy': 10000, 
            'forces': 0.04,
            'force_percent': 0.20,
            'force_threhold': 0.02
        },
        retrain_criteria = {
            'min_interval': 0, 'max_interval': 1000, 'new_samples': 5
        }, 
        restart = False, 
    )

    trainer.run()
