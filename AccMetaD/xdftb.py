"""
This module defines an ASE interface to GPC-DFTB,
Gaussian Process Corrected Density Functional Tight Binding...
QUIP-GP and DFTB+

"""

import os
import subprocess
import warnings

import numpy as np

from ase.calculators.calculator import (
    all_changes, 
    Calculator, FileIOCalculator, 
    kpts2ndarray, kpts2sizeandoffsets
)
from ase.units import Hartree, Bohr

# extended-ase
from .xquip import XQuip

ATOMIC_ENERGIES = {
    'energy': {
        'C': -38.4154,
        'O': -84.4551,
        'Pt': -63.7985,
    },
    'free_energy': {
        'C': -38.4154,
        'O': -84.6094,
        'Pt': -63.8686,
    }
}


class XDFTB(FileIOCalculator):
    """ A dftb+ calculator with ase-FileIOCalculator nomenclature
    """
    if 'DFTB_COMMAND' in os.environ:
        command = os.environ['DFTB_COMMAND'] + ' > PREFIX.out'
    else:
        command = 'dftb+ > PREFIX.out'

    implemented_properties = ['energy', 'forces', 'charges', 'stress']

    def __init__(
            self, 
            restart=None, 
            ignore_bad_restart_file=False,
            label = None,
            atoms=None, 
            kpts=None,
            run_manyDftb_steps=False, 
            slako_dir='./', 
            num_threads: int = 12, 
            # atomic energies
            binding_energy: bool = False, 
            atomic_energies: dict = {}, 
            # correction
            correction_params = None, 
            **kwargs
        ):
        """Construct a DFTB+ calculator.

        run_manyDftb_steps:  Logical
            True: many steps are run by DFTB+,
            False:a single force&energy calculation at given positions

        kpts: (int, int, int), dict, or 2D-array
            If kpts is a tuple (or list) of 3 integers, it is interpreted
            as the dimensions of a Monkhorst-Pack grid.

            If kpts is a dict, it will either be interpreted as a path
            in the Brillouin zone (*) if it contains the 'path' keyword,
            otherwise it is converted to a Monkhorst-Pack grid (**).
            (*) see ase.dft.kpoints.bandpath
            (**) see ase.calculators.calculator.kpts2sizeandoffsets

            The k-point coordinates can also be provided explicitly,
            as a (N x 3) array with the scaled coordinates (relative
            to the reciprocal unit cell vectors). Each of the N k-points
            will be given equal weight.

        ---------
        Additional object (to be set by function embed)
        pcpot: PointCharge object
            An external point charge potential (only in qmmm)
        """

        # skf tables
        #if 'DFTB_PREFIX' in os.environ:
        #    self.slako_dir = os.environ['DFTB_PREFIX'].rstrip('/') + '/'
        #else:
        #    self.slako_dir = './'
        #slako_dir = os.path.abspath(slako_dir) + '/'
        self.slako_dir = slako_dir.rstrip('/') + '/'

        self.num_threads = str(num_threads)

        # file names
        self.geo_fname = 'geo_in.gen'

        if run_manyDftb_steps:
            # minimisation of molecular dynamics is run by native DFTB+
            self.default_parameters = dict(
                Hamiltonian_='DFTB',
                Hamiltonian_SlaterKosterFiles_='Type2FileNames',
                Hamiltonian_SlaterKosterFiles_Prefix=self.slako_dir,
                Hamiltonian_SlaterKosterFiles_Separator='"-"',
                Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
                Hamiltonian_MaxAngularMomentum_='')
        else:
            # using ase to get forces and energy only
            # (single point calculation)
            self.default_parameters = dict(
                Hamiltonian_='DFTB',
                Hamiltonian_SlaterKosterFiles_='Type2FileNames',
                Hamiltonian_SlaterKosterFiles_Prefix=self.slako_dir,
                Hamiltonian_SlaterKosterFiles_Separator='"-"',
                Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
                Hamiltonian_MaxAngularMomentum_='')

        self.pcpot = None
        self.lines = None # results.tag

        self.atoms = None
        self.atoms_input = None

        self.do_forces = False
        self.outfilename = 'dftb.out'

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms,
                                  **kwargs)

        # set GP potential path
        self.correction_calculator = None
        if correction_params:
            params = correction_params.copy() # avoid destroy origin
            # get params
            correction_command = params.pop('correction_command', None)
            param_filename = params.pop('param_filename', None)
            # check params
            warnings.warn(
                'You are using DFTB with correction!', RuntimeWarning
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
        self.binding_energy = binding_energy
        if atomic_energies:
            # TODO: check energy consistent
            self.update_atomic_energies(atomic_energies)

        # Determine number of spin channels
        try:
            entry = kwargs['Hamiltonian_SpinPolarisation']
            spinpol = 'colinear' in entry.lower()
        except KeyError:
            spinpol = False
        self.nspin = 2 if spinpol else 1

        # kpoint stuff by ase
        self.kpts = kpts
        self.kpts_coord = None

        if self.kpts is not None:
            initkey = 'Hamiltonian_KPointsAndWeights'
            mp_mesh = None
            offsets = None
 
            if isinstance(self.kpts, dict):
                if 'path' in self.kpts:
                    # kpts is path in Brillouin zone
                    self.parameters[initkey + '_'] = 'Klines '
                    self.kpts_coord = kpts2ndarray(self.kpts, atoms=atoms)
                else:
                    # kpts is (implicit) definition of
                    # Monkhorst-Pack grid
                    self.parameters[initkey + '_'] = 'SupercellFolding '
                    mp_mesh, offsets = kpts2sizeandoffsets(atoms=atoms,
                                                           **self.kpts)
            elif np.array(self.kpts).ndim == 1:
                # kpts is Monkhorst-Pack grid
                self.parameters[initkey + '_'] = 'SupercellFolding '
                mp_mesh = self.kpts
                offsets = [0.] * 3
            elif np.array(self.kpts).ndim == 2:
                # kpts is (N x 3) list/array of k-point coordinates
                # each will be given equal weight
                self.parameters[initkey + '_'] = ''
                self.kpts_coord = np.array(self.kpts)
            else:
                raise ValueError('Illegal kpts definition:' + str(self.kpts))

            if mp_mesh is not None:
                eps = 1e-10
                for i in range(3):
                    key = initkey + '_empty%03d'  % i
                    val = [mp_mesh[i] if j == i else 0 for j in range(3)]
                    self.parameters[key] = ' '.join(map(str, val))
                    offsets[i] *= mp_mesh[i]
                    assert abs(offsets[i]) < eps or abs(offsets[i] - 0.5) < eps
                    # DFTB+ uses a different offset convention, where
                    # the k-point mesh is already Gamma-centered prior
                    # to the addition of any offsets
                    #if mp_mesh[i] % 2 == 0:
                    #    offsets[i] += 0.5 # use gamma-centred
                key = initkey + '_empty%03d' % 3
                self.parameters[key] = ' '.join(map(str, offsets))

            elif self.kpts_coord is not None:
                for i, c in enumerate(self.kpts_coord):
                    key = initkey + '_empty%09d'  % i
                    c_str = ' '.join(map(str, c))
                    if 'Klines' in self.parameters[initkey + '_']:
                        c_str = '1 ' + c_str
                    else:
                        c_str += ' 1.0'
                    self.parameters[key] = c_str

    def write_dftb_in(self, filename):
        """ Write the innput file for the dftb+ calculation.
            Geometry is taken always from the file 'geo_end.gen'.
        """

        outfile = open(filename, 'w')
        outfile.write('Geometry = GenFormat { \n')
        #outfile.write('    <<< "geo_end.gen" \n')
        outfile.write('  <<< %s \n' %self.geo_fname)
        outfile.write('} \n')
        outfile.write(' \n')

        params = self.parameters.copy()

        s = 'Hamiltonian_MaxAngularMomentum_'
        for key in params:
            if key.startswith(s) and len(key) > len(s):
                break
        else:
            # User didn't specify max angular mometa.  Get them from
            # the .skf files:
            symbols = set(self.atoms.get_chemical_symbols())
            for symbol in symbols:
                path = os.path.join(self.slako_dir,
                                    '{0}-{0}.skf'.format(symbol))
                l = read_max_angular_momentum(path)
                params[s + symbol] = '"{}"'.format('spdf'[l])

        # --------MAIN KEYWORDS-------
        previous_key = 'dummy_'
        myspace = ' '
        for key, value in sorted(params.items()):
            current_depth = key.rstrip('_').count('_')
            previous_depth = previous_key.rstrip('_').count('_')
            for my_backsclash in reversed(
                    range(previous_depth - current_depth)):
                outfile.write(3 * (1 + my_backsclash) * myspace + '} \n')
            outfile.write(3 * current_depth * myspace)
            if key.endswith('_') and len(value) > 0:
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' = ' + str(value) + '{ \n')
            elif (key.endswith('_') and (len(value) == 0) 
                  and current_depth == 0):  # E.g. 'Options {'
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' ' + str(value) + '{ \n')
            elif (key.endswith('_') and (len(value) == 0) 
                  and current_depth > 0):  # E.g. 'Hamiltonian_Max... = {'
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' = ' + str(value) + '{ \n')
            elif key.count('_empty') == 1:
                outfile.write(str(value) + ' \n')
            elif ((key == 'Hamiltonian_ReadInitialCharges') and 
                  (str(value).upper() == 'YES')):
                f1 = os.path.isfile(self.directory + os.sep + 'charges.dat')
                f2 = os.path.isfile(self.directory + os.sep + 'charges.bin')
                if not (f1 or f2):
                    print('charges.dat or .bin not found, switching off guess')
                    value = 'No'
                outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
            else:
                outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
            # point
            if self.pcpot is not None and ('DFTB' in str(value)):
                outfile.write('   ElectricField = { \n')
                outfile.write('      PointCharges = { \n')
                outfile.write(
                    '         CoordsAndCharges [Angstrom] = DirectRead { \n')
                outfile.write('            Records = ' +
                              str(len(self.pcpot.mmcharges)) + ' \n')
                outfile.write(
                    '            File = "dftb_external_charges.dat" \n')
                outfile.write('         } \n')
                outfile.write('      } \n')
                outfile.write('   } \n')
            previous_key = key

        current_depth = key.rstrip('_').count('_')
        for my_backsclash in reversed(range(current_depth)):
            outfile.write(3 * my_backsclash * myspace + '} \n')
        #outfile.write('ParserOptions { \n')
        #outfile.write('   IgnoreUnprocessedNodes = Yes  \n')
        #outfile.write('} \n')
        #if self.do_forces:
        #    outfile.write('Analysis { \n')
        #    outfile.write('   CalculateForces = Yes  \n')
        #    outfile.write('} \n')

        outfile.close()

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        return changed_parameters

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore unit cell for molecules:
        if not atoms.pbc.any() and 'cell' in system_changes:
            system_changes.remove('cell')
        if self.pcpot and self.pcpot.mmpositions is not None:
            system_changes.append('positions')
        # print('check !!!', system_changes)
        return system_changes

    def write_input(self, atoms, properties=None, system_changes=None):
        from ase.io import write
        # print("Calculated Properties: ...", properties)
        # if properties is not None:
        #     if 'forces' in properties or 'stress' in properties:
        #         self.do_forces = True
        self.do_forces = True
        FileIOCalculator.write_input(
            self, atoms, properties, system_changes)
        self.write_dftb_in(os.path.join(self.directory, 'dftb_in.hsd'))
        write(os.path.join(self.directory, self.geo_fname), atoms)
        # self.atoms is none until results are read out,
        # then it is set to the ones at writing input
        self.atoms_input = atoms
        self.atoms = None
        # jx: !!!
        if self.pcpot:
            self.pcpot.write_mmcharges('dftb_external_charges.dat')

        return 

    def calculate(self, atoms=None, properties=['energy'],
                    system_changes=all_changes):
        #print(self.command)
        # set threads
        os.environ['OMP_NUM_THREADS'] = self.num_threads

        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        if self.command is None:
            raise CalculatorSetupError(
                'Please set ${} environment variable '
                .format('ASE_' + self.name.upper() + '_COMMAND') +
                'or supply the command keyword')
        command = self.command
        if 'PREFIX' in command:
            command = command.replace('PREFIX', self.prefix)

        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            # Actually this may never happen with shell=True, since
            # probably the shell launches successfully.  But we soon want
            # to allow calling the subprocess directly, and then this
            # distinction (failed to launch vs failed to run) is useful.
            msg = 'Failed to execute "{}"'.format(command)
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        # remove this env, avoid affecting other codes
        os.environ.pop('OMP_NUM_THREADS', None)

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Calculator "{}" failed with command "{}" failed in '
                    '{} with error code {}'.format(self.name, command,
                                                    path, errorcode))
            raise CalculationFailed(msg)

        # read dftb results
        self.read_results()

        # after correction, the energy should be binding energy in eV
        if self.correction_calculator:
            quip_atoms = self.atoms.copy()
            quip_atoms.set_calculator(self.correction_calculator)

            error_results = {}
            error_results['energy'] = quip_atoms.get_potential_energy()
            error_results['free_energy'] = quip_atoms.get_potential_energy(
                force_consistent=True
            )
            error_results['forces'] = quip_atoms.get_forces()
            error_results['stress'] = quip_atoms.get_stress(voigt=True)

            # error predictions
            self.results['local_gap_variance'] = \
                quip_atoms.calc.results['local_gap_variance']
            self.results['gap_variance_gradient'] = \
                quip_atoms.calc.results['gap_variance_gradient']

            # add corrections
            symbols = quip_atoms.get_chemical_symbols()
            for sym in symbols:
                self.results['energy'] -= self.atomic_energies[sym]
                self.results['free_energy'] -= self.atomic_free_energies[sym]
            self.results['energy'] += error_results['energy']
            self.results['free_energy'] += error_results['free_energy']

            self.results['forces'] += error_results['forces']

            self.results['stress'] += error_results['stress']
        else:
            if self.binding_energy:
                symbols = self.atoms.get_chemical_symbols()
                for sym in symbols:
                    self.results['energy'] -= self.atomic_energies[sym]
                    self.results['free_energy'] -= self.atomic_free_energies[sym]

        return

    def read_results(self):
        """ all results are read from results.tag file
            It will be destroyed after it is read to avoid
            reading it once again after some runtime error """

        myfile = open(os.path.join(self.directory, 'results.tag'), 'r')
        self.lines = myfile.readlines()
        myfile.close()

        # print('atoms before read', self.atoms)
        # print('atoms_input before read', self.atoms_input)

        self.atoms = self.atoms_input

        charges, energy, free_energy = self.read_charges_and_energy()
        if charges is not None:
            self.results['charges'] = charges

        self.results['energy'] = energy
        self.results['free_energy'] = free_energy

        if self.do_forces:
            forces = self.read_forces()
            self.results['forces'] = forces

        self.mmpositions = None

        # stress stuff begins
        sstring = 'stress'
        have_stress = False
        stress = list()
        for iline, line in enumerate(self.lines):
            if sstring in line:
                have_stress = True
                start = iline + 1
                end = start + 3
                for i in range(start, end):
                    cell = [float(x) for x in self.lines[i].split()]
                    stress.append(cell)
        if have_stress:
            stress = -np.array(stress) * Hartree / Bohr**3
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        # stress stuff ends

        # TODO: these two seem wrong with DFTB+ master but compatible with 19.1
        # eigenvalues and fermi levels
        #fermi_levels = self.read_fermi_levels()
        #if fermi_levels is not None:
        #    self.results['fermi_levels'] = fermi_levels
        #
        #eigenvalues = self.read_eigenvalues()
        #if eigenvalues is not None:
        #    self.results['eigenvalues'] = eigenvalues

        # calculation was carried out with atoms written in write_input
        os.remove(os.path.join(self.directory, 'results.tag'))

        return

    def read_forces(self):
        #"""Read Forces from dftb output file (results.tag)."""
        """
        Read Forces from dftb output file (detailed.out).
        It seems there is no information in results.tag when SCC is not converged.
        """

        from ase.units import Hartree, Bohr

        myfile = open(os.path.join(self.directory, 'detailed.out'), 'r')
        self.lines = myfile.readlines()
        myfile.close()

        # Force line indexes
        for iline, line in enumerate(self.lines):
            fstring = 'Total Forces'
            if line.find(fstring) >= 0:
                index_force_begin = iline + 1
                index_force_end = iline + 1 + len(self.atoms)
                break

        gradients = []
        for j in range(index_force_begin, index_force_end):
            word = self.lines[j].split()
            gradients.append([float(word[k]) for k in range(1, 4)])

        return np.array(gradients) * Hartree / Bohr

    def read_charges_and_energy(self):
        """Get partial charges on atoms
            in case we cannot find charges they are set to None
        """
        infile = open(os.path.join(self.directory, 'detailed.out'), 'r')
        lines = infile.readlines()
        infile.close()

        #for line in lines:
        #    if line.strip().startswith('Total energy:'):
        #        energy = float(line.split()[2]) * Hartree
        #        break

        # for finite-temperature DFT, 0K energy is needed
        for line in lines:
            if line.strip().startswith('Extrapolated to 0:'):
                energy = float(line.split()[3]) * Hartree
                break

        # for hellman-feynman force, need force-related free energy
        for line in lines:
            if line.strip().startswith('Force related energy:'):
                free_energy = float(line.split()[3]) * Hartree
                break

        qm_charges = []
        for n, line in enumerate(lines):
            if ('Atom' and 'Charge' in line):
                chargestart = n + 1
                break
        else:
            # print('Warning: did not find DFTB-charges')
            # print('This is ok if flag SCC=No')
            return None, energy

        lines1 = lines[chargestart:(chargestart + len(self.atoms))]
        for line in lines1:
            qm_charges.append(float(line.split()[-1]))

        return np.array(qm_charges), energy, free_energy

    def get_charges(self, atoms):
        """ Get the calculated charges
        this is inhereted to atoms object """
        if 'charges' in self.results:
            return self.results['charges']
        else:
            return None

    def read_eigenvalues(self):
        """ Read Eigenvalues from dftb output file (results.tag).
            Unfortunately, the order seems to be scrambled. """
        # Eigenvalue line indexes
        index_eig_begin = None
        for iline, line in enumerate(self.lines):
            fstring = 'eigenvalues   '
            if line.find(fstring) >= 0:
                index_eig_begin = iline + 1
                line1 = line.replace(':', ',')
                ncol, nband, nkpt, nspin = map(int, line1.split(',')[-4:])
                break
        else:
            return None

        # Take into account that the last row may lack 
        # columns if nkpt * nspin * nband % ncol != 0
        nrow = int(np.ceil(nkpt * nspin * nband * 1. / ncol))
        index_eig_end = index_eig_begin + nrow
        ncol_last = len(self.lines[index_eig_end - 1].split())
        self.lines[index_eig_end - 1] += ' 0.0 ' * (ncol - ncol_last)

        eig = np.loadtxt(self.lines[index_eig_begin:index_eig_end]).flatten()
        eig *= Hartree
        N = nkpt * nband
        eigenvalues = [eig[i * N:(i + 1) * N].reshape((nkpt, nband))
                       for i in range(nspin)]

        return eigenvalues

    def read_fermi_levels(self):
        """ Read Fermi level(s) from dftb output file (results.tag). """
        # Fermi level line indexes
        for iline, line in enumerate(self.lines):
            fstring = 'fermi_level   '
            if line.find(fstring) >= 0:
                index_fermi = iline + 1
                break
        else:
            return None

        fermi_levels = []
        words = self.lines[index_fermi].split()
        assert len(words) == 2

        for word in words:
            e = float(word)
            if abs(e) > 1e-8:
                # Without spin polarization, one of the Fermi 
                # levels is equal to 0.000000000000000E+000    
                fermi_levels.append(e)

        return np.array(fermi_levels) * Hartree

    def get_ibz_k_points(self):
        return self.kpts_coord.copy()

    def get_number_of_spins(self):
        return self.nspin

    def get_eigenvalues(self, kpt=0, spin=0): 
        return self.results['eigenvalues'][spin][kpt].copy()

    def get_fermi_levels(self):
        return self.results['fermi_levels'].copy()

    def get_fermi_level(self):
        return max(self.get_fermi_levels())

    def embed(self, mmcharges=None, directory='./'):
        """Embed atoms in point-charges (mmcharges)
        """
        self.pcpot = PointChargePotential(mmcharges, self.directory)
        return self.pcpot

    # for correction
    def update_atomic_energies(self, atomic_energies=ATOMIC_ENERGIES):
        """"""
        self.atomic_energies = ATOMIC_ENERGIES['energy']
        self.atomic_free_energies = ATOMIC_ENERGIES['free_energy']

        return


class PointChargePotential:
    def __init__(self, mmcharges, directory='./'):
        """Point-charge potential for DFTB+.
        """
        self.mmcharges = mmcharges
        self.directory = directory
        self.mmpositions = None
        self.mmforces = None

    def set_positions(self, mmpositions):
        self.mmpositions = mmpositions

    def set_charges(self, mmcharges):
        self.mmcharges = mmcharges

    def write_mmcharges(self, filename='dftb_external_charges.dat'):
        """ mok all
        write external charges as monopoles for dftb+.

        """
        if self.mmcharges is None:
            print("DFTB: Warning: not writing exernal charges ")
            return
        charge_file = open(os.path.join(self.directory, filename), 'w')
        for [pos, charge] in zip(self.mmpositions, self.mmcharges):
            [x, y, z] = pos
            charge_file.write('%12.6f %12.6f %12.6f %12.6f \n'
                              % (x, y, z, charge))
        charge_file.close()

    def get_forces(self, calc, get_forces=True):
        """ returns forces on point charges if the flag get_forces=True """
        if get_forces:
            return self.read_forces_on_pointcharges()
        else:
            return np.zeros_like(self.mmpositions)

    def read_forces_on_pointcharges(self):
        """Read Forces from dftb output file (results.tag)."""
        from ase.units import Hartree, Bohr
        infile = open(os.path.join(self.directory, 'detailed.out'), 'r')
        lines = infile.readlines()
        infile.close()

        external_forces = []
        for n, line in enumerate(lines):
            if ('Forces on external charges' in line):
                chargestart = n + 1
                break
        else:
            raise RuntimeError(
                'Problem in reading forces on MM external-charges')
        lines1 = lines[chargestart:(chargestart + len(self.mmcharges))]
        for line in lines1:
            external_forces.append(
                [float(i) for i in line.split()])
        return np.array(external_forces) * Hartree / Bohr



def read_max_angular_momentum(path):
    """Read maximum angular momentum from .skf file.

    See dftb.org for A detailed description of the Slater-Koster file format.
    """
    with open(path, 'r') as fd:
        line = fd.readline()
        if line[0] == '@':
            # Extended format
            fd.readline()
            l = 3
            pos = 9
        else:
            # Simple format:
            l = 2
            pos = 7

        # Sometimes there ar commas, sometimes not:
        line = fd.readline().replace(',', ' ')

        occs = [float(f) for f in line.split()[pos:pos + l + 1]]
        for f in occs:
            if f > 0.0:
                return l
            l -= 1
