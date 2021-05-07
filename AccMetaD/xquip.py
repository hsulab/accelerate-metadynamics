import os
import warnings
import subprocess

import numpy as np

from ase.calculators.calculator import (Calculator, FileIOCalculator, all_changes)
from ase.units import Hartree, Bohr

#
from ase.io import write, read


USE_COMMAND_QUIP = False
try:
    from quippy.potential import Potential
except ImportError:
    USE_COMMAND_QUIP = True

if USE_COMMAND_QUIP:
    warnings.warn(
        'You will directly call quip in shell to calculate GP.', RuntimeWarning
    )
else:
    warnings.warn(
        'You are using Potential in quippy module.', RuntimeWarning
    )


class XQuip(FileIOCalculator):
    """"""

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    default_parameters = {'E': True, 'F': True}

    def __init__(
            self, 
            restart=None, 
            ignore_bad_restart_file=False,
            atoms=None, 
            command = None,
            param_filename = None,
            calc_args = None, # 'local_gap_variance'
            quip_xyz_name = 'quip', 
            **kwargs
        ):

        self.atoms = None

        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file,
            atoms,
            **kwargs
        )
        # TODO: sth wrong with self.directory

        if not os.path.exists(param_filename):
            raise ValueError('File Not Found %s' %(param_filename))
        self.param_filename = param_filename

        self.calc_args = calc_args
        if USE_COMMAND_QUIP:
            self.quip_command = command
            self.in_xyz = os.path.join(self.directory, quip_xyz_name+'_in.xyz')
            self.out_xyz = os.path.join(self.directory, quip_xyz_name+'_out.xyz')
        else:
            # TODO: check calc_args
            self.quip_calculator = Potential(
                param_filename = param_filename,
                calc_args = calc_args
            )

        return

    def calculate(self, atoms=None, properties=['energy'],
                    system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.results = {}
        if USE_COMMAND_QUIP:
            atoms = self.call_quip()
            self.results = {}
            self.results['energy'] = atoms.get_potential_energy()
            #self.results['free_energy'] = atoms.get_potential_energy(force_consistent=True)
            # TODO: quip not return free energy
            self.results['free_energy'] = atoms.get_potential_energy()

            self.results['forces'] = atoms.arrays['force']
            stress = -atoms.info['virial'].copy() / atoms.get_volume()
            # convert to 6-element array in Voigt order
            self.results['stress'] = np.array([
                stress[0, 0], stress[1, 1], stress[2, 2],
                stress[1, 2], stress[0, 2], stress[0, 1]
            ])
        else:
            atoms = self.atoms.copy()
            atoms.set_calculator(self.quip_calculator)

            self.results['energy'] = atoms.get_potential_energy()
            self.results['free_energy'] = atoms.get_potential_energy(force_consistent=True)
            self.results['forces'] = atoms.get_forces()
            self.results['stress'] = atoms.get_stress(voigt=True)

        if self.calc_args == 'local_gap_variance':
            self.results['local_gap_variance'] = \
                atoms.arrays['local_gap_variance']
            self.results['gap_variance_gradient'] = \
                atoms.arrays['gap_variance_gradient']

        return

    #TODO: check calc_args
    def call_quip(self):
        """"""
        # remove old files
        if os.path.exists(self.in_xyz):
            os.remove(self.in_xyz)

        if os.path.exists(self.out_xyz):
            os.remove(self.out_xyz)

        # write input
        write(self.in_xyz, self.atoms.copy()) # avoid previous results

        # calculate
        command = (
            "%s" %self.quip_command + 
            " E F V" + # energy force virial
            " atoms_filename=%s" %(os.path.basename(self.in_xyz)) + 
            " param_filename=%s" %(self.param_filename) + 
            " calc_args=\"%s\"" %self.calc_args + 
            " |" + 
            " grep AT | sed 's/AT//' > %s" %(os.path.basename(self.out_xyz))
        )

        proc = subprocess.Popen(command, shell=True, cwd=self.directory)

        errorcode = proc.wait()
        if errorcode:
            path = os.path.abspath(workDirPath)
            msg = ('Failed with command "{}" failed in '
                   '{} with error code {}'.format(command, path, errorcode))

            raise ValueError(msg)

        quip_atoms = read(self.out_xyz)

        os.remove(self.in_xyz)
        os.remove(self.in_xyz+'.idx')
        os.remove(self.out_xyz)
        
        return quip_atoms


if __name__ == '__main__':
    atoms = read('test.xyz')
    calc = QuipGap()
    atoms.set_calculator(calc)
    print(atoms)
