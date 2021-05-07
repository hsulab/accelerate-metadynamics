"""
This module defines an ASE interface to EMT with Delta Machine Learning.
"""

import os
import time
import subprocess
import warnings

import numpy as np

from ase.io import read, write

from ase.calculators.calculator import (
    all_changes, 
    Calculator, 
    FileIOCalculator
)

from ase.calculators.emt import EMT

# extended-ase
from .xquip import XQuip


ATOMIC_ENERGIES = {
    'energy': {
        'C': 3.5,
        'O': 4.6,
        'Pt': 5.85,
    },
    'free_energy': {
        'C': 3.5,
        'O': 4.6,
        'Pt': 5.85,
    },
}


class XEMT(FileIOCalculator):
    """ A Gaussian-Process corrected EMT calculator... (delta machine learning)
    """

    #implemented_properties = ['energy', 'energies', 'forces', 'stress']
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
            self, 
            correction_params = None, 
            atomicEnergies: dict = {},
            **kwargs
        ):
        """Construct a D-EMT calculator.

        """

        FileIOCalculator.__init__(
            self, 
            **kwargs
        )

        # set GP potential path
        self.correction_calculator = None
        if correction_params:
            params = correction_params.copy() # avoid destroy origin
            # get params
            correction_command = params.pop('correction_command', None)
            param_filename = params.pop('param_filename', None)
            # check params
            warnings.warn(
                'You are using EMT with correction!', RuntimeWarning
            )
            if correction_command:
                pass
            else:
                raise ValueError('No correction_command')

            if param_filename:
                if os.path.exists(param_filename):
                    param_filename = os.path.abspath(param_filename)
                else:
                    raise ValueError('%s not exists.' %param_filename)
            else:
                raise ValueError('No param_filename')

            # generate calculator
            self.correction_calculator = XQuip(
                directory = self.directory, 
                command = correction_command,
                param_filename = param_filename, 
                **params, 
            )
        else:
            pass

        # set atomic energies
        self.update_atomic_energies()

        return

    def calculate(self, atoms=None, properties=['energy'],
                    system_changes=all_changes):
        # directories are created automatically.
        Calculator.calculate(self, atoms, properties, system_changes)

        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # EMT
        #start_time = time.time()
        emtAtoms = self.atoms.copy()
        emtAtoms.set_calculator(EMT())
        #end_time = time.time()
        #print(
        #    '%s cost time: %.3f s' % ('emt in xemt', end_time-start_time)
        #)

        self.results['energy'] = emtAtoms.get_potential_energy()
        self.results['free_energy'] = emtAtoms.get_potential_energy(force_consistent=True)
        self.results['forces'] = emtAtoms.get_forces()
        self.results['stress'] = emtAtoms.get_stress(voigt=True)

        # after correction, the energy should be binding energy in eV
        if self.correction_calculator:
            quipAtoms = self.atoms.copy()
            quipAtoms.set_calculator(self.correction_calculator)

            error_results = {}
            error_results['energy'] = quipAtoms.get_potential_energy()
            error_results['free_energy'] = quipAtoms.get_potential_energy(
                force_consistent=True
            )
            error_results['forces'] = quipAtoms.get_forces()
            error_results['stress'] = quipAtoms.get_stress(voigt=True)

            # error predictions
            self.results['local_gap_variance'] = \
                quipAtoms.calc.results['local_gap_variance']
            self.results['gap_variance_gradient'] = \
                quipAtoms.calc.results['gap_variance_gradient']

            # add corrections
            symbols = self.atoms.get_chemical_symbols()
            for sym in symbols:
                self.results['energy'] -= self.atomic_energies[sym]
                self.results['free_energy'] -= self.atomic_free_energies[sym]
            self.results['energy'] += error_results['energy']
            self.results['free_energy'] += error_results['free_energy']

            self.results['forces'] += error_results['forces']

            self.results['stress'] += error_results['stress']
        else:
            symbols = self.atoms.get_chemical_symbols()
            for sym in symbols:
                self.results['energy'] -= self.atomic_energies[sym]
                self.results['free_energy'] -= self.atomic_free_energies[sym]

        return

    def update_atomic_energies(self, atomic_energies=ATOMIC_ENERGIES):
        """"""
        self.atomic_energies = ATOMIC_ENERGIES['energy']
        self.atomic_free_energies = ATOMIC_ENERGIES['free_energy']

        return

if __name__ == '__main__':
    from ase import Atoms
    # test
    print('USE_COMMAND_QUIP ', USE_COMMAND_QUIP)
    atoms = Atoms(
        'Pt', 
        positions=[(0.,0.,0.)],
        cell = [[10.,0.,0.],[0.,10.,0.],[0.,0.,10.]]
    )
    atoms.set_calculator(XEMT())
    print(atoms.get_potential_energy())
    print(atoms.get_potential_energy(force_consistent=True))

